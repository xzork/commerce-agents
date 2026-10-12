# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The opt-in account-status and acting-customer systems: their tools, the customer
gates, the account grounding rule, and the prompt under each switch."""

from __future__ import annotations

import pytest

from commerce_common.grounding import first_forced_tool
from commerce_common.memory import InMemoryMemoryStore
from shopping_agent import AccountStatus, CustomerSummary, UserPreferences
from shopping_agent.executor import ShoppingToolExecutor, build_memory
from shopping_agent.gates import (
    CUSTOMER_GATE,
    CUSTOMER_PROVENANCE_GATE,
    CUSTOMER_SCOPED_TOOLS,
    customer_provenance_error,
)
from shopping_agent.grounding import GROUNDING_RULES
from shopping_agent.prompt import build_dynamic_context, build_static_system
from shopping_agent.tools.registry import build_tools

STATUSES = [
    AccountStatus(
        account_name="Kestrel Hardware",
        balance=5838.04,
        balance_due_date="2026-10-20",
        term_days=13,
        due_date="2026-10-25",
        currency="TRY",
    ),
    AccountStatus(account_name="Kestrel Hardware Annex", balance=0.0, currency="TRY"),
]
CUSTOMERS = [
    CustomerSummary(customer_id="c-41", name="Kestrel Hardware", city="Springfield"),
    CustomerSummary(customer_id="c-42", name="Kestrel Lighting", tax_number_masked="38******67"),
]
ACCOUNT_TOOLS = {"get_account_status"}
ACTING_TOOLS = {"find_customers", "select_customer"}


@pytest.fixture
def accounts(backend, monkeypatch):
    """The fake backend with the optional account and customer methods implemented."""
    selected: list[str] = []

    async def get_account_status(session):
        return STATUSES

    async def find_customers(session, query, limit=8):
        return CUSTOMERS[:limit]

    async def select_customer(session, customer_id):
        selected.append(customer_id)

    monkeypatch.setattr(backend, "get_account_status", get_account_status)
    monkeypatch.setattr(backend, "find_customers", find_customers)
    monkeypatch.setattr(backend, "select_customer", select_customer)
    return selected


def executor_for(backend, config, skills, session, state, **switches):
    return ShoppingToolExecutor(
        backend=backend,
        config=config.model_copy(update=switches),
        skills=skills,
        session=session,
        state=state,
        memory=build_memory(config, InMemoryMemoryStore()),
    )


def names(config) -> set[str]:
    return {tool["name"] for tool in build_tools(config, [])}


# -- the switches ----------------------------------------------------------------------


def test_both_systems_are_off_by_default(config):
    assert not (ACCOUNT_TOOLS | ACTING_TOOLS) & names(config)


def test_each_switch_adds_only_its_own_tools(config):
    baseline = names(config)
    assert names(config.model_copy(update={"enable_account_status": True})) - baseline == (
        ACCOUNT_TOOLS
    )
    assert names(config.model_copy(update={"enable_acting_customer": True})) - baseline == (
        ACTING_TOOLS
    )


async def test_a_switched_off_tool_answers_as_absent(backend, config, skills, session, state):
    executor = executor_for(backend, config, skills, session, state)
    result = await executor.execute("get_account_status", {})
    assert result.is_error
    assert "not something this store offers" in result.result_text


def test_staff_status_lines_name_the_staff_member(config):
    def status_reader(cfg) -> str:
        search = next(t for t in build_tools(cfg, []) if t["name"] == "search_products")
        return search["input_schema"]["properties"]["status"]["description"]

    assert "the customer sees" in status_reader(config)
    staff = config.model_copy(update={"enable_acting_customer": True})
    assert "the staff member sees" in status_reader(staff)


# -- account status --------------------------------------------------------------------


async def test_account_status_passes_the_backends_values_through(
    backend, config, skills, session, state, accounts
):
    executor = executor_for(backend, config, skills, session, state, enable_account_status=True)
    result = await executor.execute("get_account_status", {})
    assert not result.is_error
    for value in ("Kestrel Hardware Annex", "5838.04", "2026-10-20", '"term_days": 13', "TRY"):
        assert value in result.result_text


async def test_account_status_unavailable_gives_no_figures(
    backend, config, skills, session, state, monkeypatch
):
    async def down(session):
        raise ConnectionError("accounting system timed out")

    monkeypatch.setattr(backend, "get_account_status", down)
    executor = executor_for(backend, config, skills, session, state, enable_account_status=True)
    result = await executor.execute("get_account_status", {})
    assert result.is_error
    assert result.result_text == executor.unavailable_text.format(name="get_account_status")


async def test_a_backend_without_account_status_reads_as_unavailable(
    backend, config, skills, session, state
):
    executor = executor_for(backend, config, skills, session, state, enable_account_status=True)
    result = await executor.execute("get_account_status", {})
    assert result.is_error
    assert "temporarily unavailable" in result.result_text


def test_a_balance_question_grounds_through_account_status(config, state):
    on = config.model_copy(update={"enable_account_status": True})
    question = "What is my balance?"
    assert first_forced_tool(GROUNDING_RULES, on, question, state) == "get_account_status"
    assert first_forced_tool(GROUNDING_RULES, config, question, state) is None


def test_the_account_rule_waits_for_a_selected_customer(config, state):
    staff = config.model_copy(
        update={"enable_account_status": True, "enable_acting_customer": True}
    )
    question = "How much is outstanding?"
    assert first_forced_tool(GROUNDING_RULES, staff, question, state) is None
    state.remember_customers(CUSTOMERS)
    state.acting_customer_id = "c-41"
    assert first_forced_tool(GROUNDING_RULES, staff, question, state) == "get_account_status"


# -- acting customer -------------------------------------------------------------------


async def test_a_customer_from_the_results_is_selected(
    backend, config, skills, session, state, accounts
):
    executor = executor_for(backend, config, skills, session, state, enable_acting_customer=True)
    found = await executor.execute("find_customers", {"query": "kestrel"})
    assert "c-42" in found.result_text and "38******67" in found.result_text

    result = await executor.execute("select_customer", {"customer_id": "c-42"})
    assert result.result_text == "Now acting for customer c-42."
    assert state.acting_customer_id == "c-42"
    assert accounts == ["c-42"]


async def test_an_invented_customer_id_is_refused(
    backend, config, skills, session, state, accounts
):
    executor = executor_for(backend, config, skills, session, state, enable_acting_customer=True)
    await executor.execute("find_customers", {"query": "kestrel"})
    result = await executor.execute("select_customer", {"customer_id": "c-99"})
    assert result.blocked == CUSTOMER_PROVENANCE_GATE
    assert result.result_text == customer_provenance_error("c-99")
    assert state.acting_customer_id is None
    assert accounts == []


@pytest.mark.parametrize("tool", sorted(CUSTOMER_SCOPED_TOOLS))
async def test_customer_scoped_tools_wait_for_a_selection(
    backend, config, skills, session, state, accounts, tool
):
    executor = executor_for(
        backend,
        config,
        skills,
        session,
        state,
        enable_acting_customer=True,
        enable_account_status=True,
    )
    result = await executor.execute(tool, {})
    assert result.blocked == CUSTOMER_GATE
    assert "find_customers" in result.result_text


async def test_the_gate_is_off_without_acting_customer(backend, config, skills, session, state):
    executor = executor_for(backend, config, skills, session, state)
    result = await executor.execute("get_cart", {})
    assert result.blocked is None and not result.is_error


async def test_switching_customer_empties_the_cart(
    backend, config, skills, session, state, accounts
):
    executor = executor_for(backend, config, skills, session, state, enable_acting_customer=True)
    await executor.execute("find_customers", {"query": "kestrel"})
    await executor.execute("select_customer", {"customer_id": "c-41"})
    await executor.execute("search_products", {"query": "tent"})
    added = await executor.execute("add_to_cart", {"product_id": "p-100"})
    assert "Added" in added.result_text

    switched = await executor.execute("select_customer", {"customer_id": "c-42"})
    assert switched.result_text == (
        "Now acting for customer c-42. The previous customer's cart was emptied."
    )
    assert backend.cart_items == {}
    assert any(event.type == "cart_update" for event in switched.events)

    again = await executor.execute("select_customer", {"customer_id": "c-42"})
    assert again.result_text == "Already acting for customer c-42."


# -- prompt ----------------------------------------------------------------------------


def test_the_prompt_is_unchanged_with_both_switches_off(config, skills):
    text = build_static_system(config, skills)
    assert "talking with a customer inside the store's app" in text
    assert "select_customer" not in text and "get_account_status" not in text


def test_the_prompt_addresses_staff_acting_for_one_customer(config, skills):
    staff = config.model_copy(update={"enable_acting_customer": True})
    text = build_static_system(staff, skills)
    assert "talking with a member of ACME's staff" in text
    assert "Never act for or describe the account of any customer but the selected one" in text


def test_the_prompt_grounds_account_figures_when_account_status_is_on(config, skills):
    text = build_static_system(config.model_copy(update={"enable_account_status": True}), skills)
    assert "only from a get_account_status result in this conversation" in text


def test_the_context_names_the_acting_customer(state):
    state.remember_customers(CUSTOMERS)
    state.acting_customer_id = "c-41"
    block = build_dynamic_context(
        preferences=UserPreferences(user_id="staff-7"),
        memory_facts=[],
        cart=None,
        page=None,
        acting_customer=state.acting_customer,
    )
    assert '"acting_customer"' in block and "Kestrel Hardware" in block
    without = build_dynamic_context(
        preferences=UserPreferences(user_id="staff-7"), memory_facts=[], cart=None, page=None
    )
    assert "acting_customer" not in without
