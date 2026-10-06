import os
import re
import chromadb

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "company_records"
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

all_records = collection.get()
ALL_METADATA = all_records.get("metadatas", [])


def get_by_id(item_id: str):
    item_id = item_id.upper()

    if item_id.startswith("INV-"):
        chroma_id = f"inv_{item_id}"
    elif item_id.startswith("TSK-"):
        chroma_id = f"task_{item_id}"
    elif item_id.startswith("EMP"):
        chroma_id = f"emp_{item_id}"
    elif item_id.startswith("CLI"):
        chroma_id = f"cli_{item_id}"
    else:
        return None

    result = collection.get(ids=[chroma_id])
    documents = result.get("documents", [])

    if documents:
        return documents[0]

    return None


def find_employee(query: str):
    query_lower = query.lower()

    # 1. Try exact full-name matching first
    for metadata in ALL_METADATA:
        if metadata.get("entity_type") != "employee":
            continue

        name = metadata.get("name", "").strip()

        if name and name.lower() in query_lower:
            emp_id = metadata.get("emp_id")

            if emp_id:
                return get_by_id(emp_id)

    # 2. Try first-name matching
    for metadata in ALL_METADATA:
        if metadata.get("entity_type") != "employee":
            continue

        name = metadata.get("name", "").strip()

        if not name:
            continue

        first_name = name.split()[0].lower()

        if re.search(rf"\b{re.escape(first_name)}\b", query_lower):
            emp_id = metadata.get("emp_id")

            if emp_id:
                return get_by_id(emp_id)

    return None


def find_client(query: str):
    query_lower = query.lower()
    client_matches = []

    for metadata in ALL_METADATA:
        if metadata.get("entity_type") != "client":
            continue

        name = metadata.get("name", "")

        if not name:
            continue

        name_lower = name.lower()

        if name_lower in query_lower:
            client_matches.append(metadata)
            continue

        simplified_name = name_lower

        for suffix in [
            " pvt ltd",
            " private limited",
            " ltd",
            " limited",
        ]:
            simplified_name = simplified_name.replace(suffix, "")

        simplified_name = simplified_name.strip()

        if simplified_name and simplified_name in query_lower:
            client_matches.append(metadata)

    if client_matches:
        client_matches.sort(
            key=lambda x: x.get("client_id", "")
        )

        client_id = client_matches[0].get("client_id")

        if client_id:
            return get_by_id(client_id)

    return None


def find_exact_id(query: str):
    patterns = [
        r"INV-\d{4}",
        r"TSK-\d{4}",
        r"EMP\d{3}",
        r"CLI\d{3}",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            query,
            re.IGNORECASE
        )

        if match:
            item_id = match.group(0).upper()
            document = get_by_id(item_id)

            if document:
                return document

    return None


def get_records_by_metadata(entity_type=None, **filters):
    records = []

    for metadata in ALL_METADATA:
        if entity_type is not None:
            if metadata.get("entity_type") != entity_type:
                continue

        matched = True

        for key, value in filters.items():
            if metadata.get(key) != value:
                matched = False
                break

        if matched:
            records.append(metadata)

    return records


def metadata_to_documents(records):
    documents = []

    for metadata in records:
        entity_type = metadata.get("entity_type")

        if entity_type == "employee":
            item_id = metadata.get("emp_id")
        elif entity_type == "client":
            item_id = metadata.get("client_id")
        elif entity_type == "invoice":
            item_id = metadata.get("invoice_id")
        elif entity_type == "task":
            item_id = metadata.get("task_id")
        else:
            continue

        document = get_by_id(item_id)

        if document:
            documents.append(document)

    return documents


def find_department(query):
    q = query.lower()

    departments = [
        "engineering",
        "sales",
        "marketing",
        "design",
        "accounts",
        "finance",
        "operations",
        "support",
        "hr",
    ]

    for department in departments:
        if department in q:
            return department.title()

    return None


def structured_search(query):
    q = query.lower()

    if (
        "blocked task" in q
        or "blocked tasks" in q
        or "blocked kaam" in q
    ):
        records = get_records_by_metadata(
            entity_type="task",
            status="Blocked"
        )

        return metadata_to_documents(records)

    if (
        "overdue invoice" in q
        or "overdue invoices" in q
        or ("overdue" in q and "invoice" in q)
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Overdue"
        )

        return metadata_to_documents(records)

    if (
        "pending invoice" in q
        or "pending invoices" in q
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Pending"
        )

        return metadata_to_documents(records)

    if (
        "paid invoice" in q
        or "paid invoices" in q
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Paid"
        )

        return metadata_to_documents(records)

    department = find_department(query)

    if department:
        records = get_records_by_metadata(
            entity_type="employee",
            department=department
        )

        if records:
            return metadata_to_documents(records)

    return None


def is_count_query(query):
    q = query.lower()

    count_words = [
        "kitne",
        "kitni",
        "kitna count",
        "count",
        "how many",
        "number of",
        "total number",
    ]

    return any(word in q for word in count_words)


def is_sum_query(query):
    q = query.lower()

    sum_words = [
        "total amount",
        "total kitna",
        "total kitne",
        "sum",
        "combined",
        "overall amount",
        "kul amount",
        "kul kitna",
    ]

    return any(word in q for word in sum_words)


def aggregation_search(query):
    q = query.lower()

    if (
        "overdue" in q
        and "invoice" in q
        and is_count_query(q)
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Overdue"
        )

        return f"COUNT: {len(records)} overdue invoices."

    if (
        "pending" in q
        and "invoice" in q
        and is_count_query(q)
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Pending"
        )

        return f"COUNT: {len(records)} pending invoices."

    if (
        "paid" in q
        and "invoice" in q
        and is_count_query(q)
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Paid"
        )

        return f"COUNT: {len(records)} paid invoices."

    if (
        "blocked" in q
        and "task" in q
        and is_count_query(q)
    ):
        records = get_records_by_metadata(
            entity_type="task",
            status="Blocked"
        )

        return f"COUNT: {len(records)} blocked tasks."

    if (
        "overdue" in q
        and "invoice" in q
        and is_sum_query(q)
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Overdue"
        )

        total = sum(
            metadata.get("due_amount_inr", 0)
            for metadata in records
        )

        return (
            f"TOTAL_DUE_AMOUNT: ₹{total:,} "
            f"across {len(records)} overdue invoices."
        )

    if (
        "pending" in q
        and "invoice" in q
        and is_sum_query(q)
    ):
        records = get_records_by_metadata(
            entity_type="invoice",
            status="Pending"
        )

        total = sum(
            metadata.get("due_amount_inr", 0)
            for metadata in records
        )

        return (
            f"TOTAL_DUE_AMOUNT: ₹{total:,} "
            f"across {len(records)} pending invoices."
        )

    return None


def semantic_search(query):
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    if not documents:
        return ""

    return "\n".join(documents)


def get_context(query: str):
    if not query or not query.strip():
        return ""

    document = find_exact_id(query)

    if document:
        return document

    document = find_employee(query)

    if document:
        return document

    document = find_client(query)

    if document:
        return document

    aggregated = aggregation_search(query)

    if aggregated:
        return aggregated

    documents = structured_search(query)

    if documents:
        return "\n".join(documents)

    return semantic_search(query)

