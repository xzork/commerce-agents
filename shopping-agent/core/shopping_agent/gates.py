# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The cart gates. A cart write accepts only product ids that catalog or order tools
returned this session (or lines already in the cart), holds an add of a product that
still has options to choose and points at its variants, caps the resulting line
quantity at the config's limit, and reports any cap it applied; writes for one session
are serialized because a turn's tool calls run concurrently.

With acting-customer on, the customer gates apply the same discipline to customers:
``select_customer`` accepts only an id ``find_customers`` returned this session, and
every customer-scoped tool is held until a customer is selected.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Sequence

from commerce_common.streaming import AgentEvent, ToolOutcome

from .backend import StorefrontBackend
from .config import ShoppingAgentConfig
from .fencing import STOREFRONT_FENCE
from .serialization import cart_payload, cart_summary
from .types import Cart, Order, Product, ShoppingSessionContext, ShoppingSessionState

PROVENANCE_GATE = "provenance"
OPTIONS_GATE = "options"
CUSTOMER_GATE = "customer"
CUSTOMER_PROVENANCE_GATE = "customer_provenance"

# The tools that act for a customer: with acting-customer on, each waits for a selection.
CUSTOMER_SCOPED_TOOLS = frozenset(
    {
        "get_cart",
        "add_to_cart",
        "update_cart_item",
        "remove_from_cart",
        "checkout",
        "get_orders",
        "get_order_status",
        "present_order_status",
        "get_account_status",
    }
)


def provenance_error(product_id: str) -> str:
    # get_product_details comes first in the hint: text search does not match ids, and
    # an empty search reads to the model as proof the product does not exist.
    return (
        f"product_id {product_id} was not returned by catalog or order tools in this "
        "session. Resolve it first: call get_product_details with this exact id (text "
        "search does not match product ids), or find it via search or order history, "
        "then add it using a product_id from those results."
    )


def check_provenance(state: ShoppingSessionState, product_id: str) -> ToolOutcome | None:
    """The held outcome when ``product_id`` has no session provenance, else None."""
    if product_id in state.seen_products:
        return None
    return ToolOutcome.held(PROVENANCE_GATE, provenance_error(product_id))


def options_error(product: Product) -> str:
    # Option names are catalog text arriving outside the fence: sanitized and kept
    # short. The values themselves are in the fenced record the model already holds.
    names = STOREFRONT_FENCE.sanitize_text(", ".join(product.options), max_chars=60)
    return (
        f"product_id {product.product_id} has options still to choose ({names}), so the "
        "cart takes one of its variants. Settle each option from what the customer said "
        "or the customer's profile, ask once with the values as chips when one is still "
        "open, then add the matching variant's product_id from the variants "
        "get_product_details returns for this id."
    )


def check_options(state: ShoppingSessionState, product_id: str) -> ToolOutcome | None:
    """The held outcome when the record for ``product_id`` is a family with options
    still to choose; its variants are what the cart takes."""
    product = state.seen_products.get(product_id)
    if product is None or not product.has_options:
        return None
    return ToolOutcome.held(OPTIONS_GATE, options_error(product))


def customer_gate_error(tool_name: str) -> str:
    return (
        f"{tool_name} acts for a customer, and no customer is selected yet. Ask the staff "
        "member which customer this is for, find them with find_customers, then select one "
        "with select_customer."
    )


def check_acting_customer(
    config: ShoppingAgentConfig, state: ShoppingSessionState, tool_name: str
) -> ToolOutcome | None:
    """The held outcome for a customer-scoped tool called before a customer is selected,
    when acting-customer is on; else None."""
    if (
        not config.enable_acting_customer
        or tool_name not in CUSTOMER_SCOPED_TOOLS
        or state.acting_customer_id is not None
    ):
        return None
    return ToolOutcome.held(CUSTOMER_GATE, customer_gate_error(tool_name))


def customer_provenance_error(customer_id: str) -> str:
    return (
        f"customer_id {customer_id} was not returned by find_customers in this session. "
        "Search for the customer first, then select a customer_id from those results."
    )


async def gated_select_customer(
    *,
    backend: StorefrontBackend,
    config: ShoppingAgentConfig,
    session: ShoppingSessionContext,
    state: ShoppingSessionState,
    customer_id: str,
) -> ToolOutcome:
    """Select the customer a staff user acts for. Switching from another customer
    empties the cart first, so nothing staged for them carries over."""
    if customer_id not in state.seen_customers:
        return ToolOutcome.held(CUSTOMER_PROVENANCE_GATE, customer_provenance_error(customer_id))
    if customer_id == state.acting_customer_id:
        return ToolOutcome(f"Already acting for customer {customer_id}.")

    events: list[AgentEvent] = []
    note = ""
    async with _cart_lock(session):
        if state.acting_customer_id is not None and config.enable_cart:
            cart = await backend.get_cart(session)
            if cart.items:
                for item in cart.items:
                    cart = await backend.remove_from_cart(session, item.product_id)
                events.append(AgentEvent.cart_update(cart_payload(cart)))
                note = " The previous customer's cart was emptied."
        await backend.select_customer(session, customer_id)
        state.acting_customer_id = customer_id
    # The confirmation names the id only: the name is backend text and stays in fences.
    return ToolOutcome(f"Now acting for customer {customer_id}.{note}", events)


def remember_order_items(state: ShoppingSessionState, orders: Sequence[Order]) -> None:
    """Items on the customer's own orders count as provenance, so a reorder needs no search."""
    state.remember_products(
        [
            Product(
                product_id=item.product_id,
                title=item.title,
                price=item.price,
                option_values=item.option_values,
                variant_of=item.variant_of,
            )
            for order in orders
            for item in order.items
        ]
    )


# The gates read the cart, compute, then write; a second mutation for the same session
# in the same gather must not interleave with that. Locks live only while held.
_cart_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()


def _cart_lock(session: ShoppingSessionContext) -> asyncio.Lock:
    lock = _cart_locks.get(session.session_id)
    if lock is None:
        lock = _cart_locks[session.session_id] = asyncio.Lock()
    return lock


def _written(text: str, cart: Cart) -> ToolOutcome:
    return ToolOutcome(text, [AgentEvent.cart_update(cart_payload(cart))])


async def gated_add_to_cart(
    *,
    backend: StorefrontBackend,
    config: ShoppingAgentConfig,
    session: ShoppingSessionContext,
    state: ShoppingSessionState,
    product_id: str,
    quantity: int,
) -> ToolOutcome:
    if held := check_provenance(state, product_id) or check_options(state, product_id):
        return held
    requested = max(1, quantity)
    max_quantity = config.max_quantity_per_item
    async with _cart_lock(session):
        current = await backend.get_cart(session)
        existing = next((i for i in current.items if i.product_id == product_id), None)
        if existing is None and len(current.items) >= config.max_cart_lines:
            return ToolOutcome.error("The cart is full.")
        allowed = min(requested, max(0, max_quantity - (existing.quantity if existing else 0)))
        if allowed <= 0:
            return ToolOutcome.error(
                f"This item is already at the per-item limit of {max_quantity}."
            )
        cart = await backend.add_to_cart(session, product_id, allowed)
    # The confirmation names the id only: titles are catalog text and stay inside fences.
    capped = f" (capped at the per-item limit of {max_quantity})" if allowed < requested else ""
    return _written(
        f"Added {product_id} x{allowed}{capped}. Cart now has {cart_summary(cart)}.", cart
    )


async def gated_update_cart_item(
    *,
    backend: StorefrontBackend,
    config: ShoppingAgentConfig,
    session: ShoppingSessionContext,
    state: ShoppingSessionState,
    product_id: str,
    quantity: int,
) -> ToolOutcome:
    requested = max(1, quantity)
    applied = min(requested, config.max_quantity_per_item)
    async with _cart_lock(session):
        if held := await _check_provenance_or_cart(backend, session, state, product_id):
            return held
        cart = await backend.update_cart_item(session, product_id, applied)
    capped = (
        f" (capped at the per-item limit of {config.max_quantity_per_item})"
        if applied < requested
        else ""
    )
    return _written(f"Updated quantity{capped}. Cart now has {cart_summary(cart)}.", cart)


async def gated_remove_from_cart(
    *,
    backend: StorefrontBackend,
    session: ShoppingSessionContext,
    state: ShoppingSessionState,
    product_id: str,
) -> ToolOutcome:
    async with _cart_lock(session):
        if held := await _check_provenance_or_cart(backend, session, state, product_id):
            return held
        cart = await backend.remove_from_cart(session, product_id)
    return _written(f"Removed. Cart now has {cart_summary(cart)}.", cart)


async def _check_provenance_or_cart(
    backend: StorefrontBackend,
    session: ShoppingSessionContext,
    state: ShoppingSessionState,
    product_id: str,
) -> ToolOutcome | None:
    """Update and remove also accept a line already in the cart, which may predate the
    session; the cart is fetched only when provenance alone does not pass."""
    if check_provenance(state, product_id) is None:
        return None
    current = await backend.get_cart(session)
    if any(item.product_id == product_id for item in current.items):
        return None
    return ToolOutcome.held(PROVENANCE_GATE, provenance_error(product_id))
