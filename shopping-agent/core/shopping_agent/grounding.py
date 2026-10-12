# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The shopping agent's grounding rules, in precedence order: a store-terms question
starts from search_policies, a post-purchase question from get_orders, a question about
what the customer owes from get_account_status, and a mention of an unseen product id
from get_product_details. A rule whose system the config switches off never fires, and
the account rule waits for a selected customer when a staff user acts for one."""

from __future__ import annotations

from typing import Any

from commerce_common.grounding import GroundingRule, find_token, matches_terms_and_cues

from .config import ShoppingAgentConfig
from .types import ShoppingSessionState


def _policy(
    config: ShoppingAgentConfig, text: str, _: ShoppingSessionState
) -> dict[str, Any] | None:
    fires = (
        config.enable_policies
        and config.policy_grounding_gate
        and matches_terms_and_cues(text, config.policy_intent_terms, config.policy_intent_cues)
    )
    return {} if fires else None


def _orders(
    config: ShoppingAgentConfig, text: str, _: ShoppingSessionState
) -> dict[str, Any] | None:
    fires = (
        config.enable_orders
        and config.order_grounding_gate
        and matches_terms_and_cues(text, config.order_intent_terms, config.order_intent_cues)
    )
    return {} if fires else None


def _account(
    config: ShoppingAgentConfig, text: str, state: ShoppingSessionState
) -> dict[str, Any] | None:
    fires = (
        config.enable_account_status
        and config.account_grounding_gate
        and not (config.enable_acting_customer and state.acting_customer_id is None)
        and matches_terms_and_cues(text, config.account_intent_terms, config.account_intent_cues)
    )
    return {} if fires else None


def _catalog(
    config: ShoppingAgentConfig, text: str, state: ShoppingSessionState
) -> dict[str, Any] | None:
    if not config.catalog_grounding_gate:
        return None
    token = find_token(text, config.product_id_patterns)
    # An id the session already resolved needs no forced re-read; ids compare case-insensitively.
    if token is None or token.upper() in {pid.upper() for pid in state.seen_products}:
        return None
    return {"product_id": token}


GROUNDING_RULES: tuple[GroundingRule, ...] = (
    GroundingRule("policy", "search_policies", _policy),
    GroundingRule(
        "orders",
        "get_orders",
        _orders,
        prefetch_intro=lambda _: (
            "Recent orders for this turn, fetched by the host (the same data a get_orders "
            "call returns):"
        ),
    ),
    GroundingRule(
        "account",
        "get_account_status",
        _account,
        prefetch_intro=lambda _: (
            "Account status for this turn, fetched by the host (the same data a "
            "get_account_status call returns):"
        ),
    ),
    GroundingRule(
        "catalog",
        "get_product_details",
        _catalog,
        prefetch_intro=lambda args: (
            f"Catalog record for {args['product_id']}, fetched by the host (the same data a "
            "get_product_details call returns):"
        ),
    ),
)
