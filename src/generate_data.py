import pandas as pd
import numpy as np
import random
from faker import Faker
from pathlib import Path

# --------------------------------------------------
# PULSEOPS - SYNTHETIC LOGISTICS DATA GENERATOR
# --------------------------------------------------

fake = Faker("en_AU")

# Reproducible results
np.random.seed(42)
random.seed(42)
Faker.seed(42)

# Dataset sizes
NUM_CUSTOMERS = 100_000
NUM_ORDERS = 500_000
NUM_COMPLAINTS = 30_000

# Output folder
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT"]


# --------------------------------------------------
# 1. WAREHOUSES
# --------------------------------------------------

print("Generating warehouses...")

warehouses = pd.DataFrame({
    "warehouse_id": range(1, 11),
    "warehouse_name": [
        "Sydney Central",
        "Melbourne North",
        "Brisbane South",
        "Adelaide West",
        "Perth Central",
        "Hobart Distribution",
        "Canberra Central",
        "Sydney West",
        "Melbourne East",
        "Brisbane North"
    ],
    "state": [
        "NSW",
        "VIC",
        "QLD",
        "SA",
        "WA",
        "TAS",
        "ACT",
        "NSW",
        "VIC",
        "QLD"
    ]
})

warehouses.to_csv(
    OUTPUT_DIR / "warehouses.csv",
    index=False
)


# --------------------------------------------------
# 2. CUSTOMERS
# --------------------------------------------------

print("Generating customers...")

customers = pd.DataFrame({
    "customer_id": range(1, NUM_CUSTOMERS + 1),

    "customer_name": [
        fake.name()
        for _ in range(NUM_CUSTOMERS)
    ],

    "state": np.random.choice(
        STATES,
        NUM_CUSTOMERS
    ),

    "customer_segment": np.random.choice(
        ["Consumer", "Small Business", "Enterprise"],
        NUM_CUSTOMERS,
        p=[0.70, 0.22, 0.08]
    ),

    "signup_date": [
        fake.date_between(
            start_date="-5y",
            end_date="today"
        )
        for _ in range(NUM_CUSTOMERS)
    ]
})

customers.to_csv(
    OUTPUT_DIR / "customers.csv",
    index=False
)


# --------------------------------------------------
# 3. ORDERS
# --------------------------------------------------

print("Generating 500,000 orders...")

date_range = pd.date_range(
    start="2025-01-01",
    end="2026-07-31"
)

orders = pd.DataFrame({
    "order_id": range(1, NUM_ORDERS + 1),

    "customer_id": np.random.randint(
        1,
        NUM_CUSTOMERS + 1,
        NUM_ORDERS
    ),

    "warehouse_id": np.random.randint(
        1,
        11,
        NUM_ORDERS
    ),

    "order_date": np.random.choice(
        date_range,
        NUM_ORDERS
    ),

    "order_amount": np.round(
        np.random.gamma(
            shape=2,
            scale=45,
            size=NUM_ORDERS
        ),
        2
    ),

    "order_channel": np.random.choice(
        ["Web", "Mobile App", "Marketplace", "Business Portal"],
        NUM_ORDERS,
        p=[0.40, 0.35, 0.15, 0.10]
    )
})


# --------------------------------------------------
# 4. DELIVERIES
# --------------------------------------------------

print("Generating deliveries...")

deliveries = orders[
    ["order_id", "warehouse_id", "order_date"]
].copy()

promised_days = np.random.randint(
    2,
    6,
    NUM_ORDERS
)

deliveries["promised_delivery_date"] = (
    pd.to_datetime(deliveries["order_date"])
    + pd.to_timedelta(promised_days, unit="D")
)

# Most deliveries are on time,
# but some are intentionally late.
delay_days = np.random.choice(
    [0, 1, 2, 3, 5, 7],
    NUM_ORDERS,
    p=[0.72, 0.12, 0.07, 0.05, 0.03, 0.01]
)

deliveries["actual_delivery_date"] = (
    deliveries["promised_delivery_date"]
    + pd.to_timedelta(delay_days, unit="D")
)

deliveries["delivery_cost"] = np.round(
    np.random.uniform(
        5,
        40,
        NUM_ORDERS
    ),
    2
)

deliveries["delivery_status"] = np.where(
    deliveries["actual_delivery_date"]
    > deliveries["promised_delivery_date"],
    "Late",
    "On Time"
)


# --------------------------------------------------
# 5. COMPLAINTS
# --------------------------------------------------

print("Generating complaints...")

complaint_orders = orders.sample(
    n=NUM_COMPLAINTS,
    random_state=42
)

complaints = pd.DataFrame({
    "complaint_id": range(
        1,
        NUM_COMPLAINTS + 1
    ),

    "order_id": complaint_orders[
        "order_id"
    ].values,

    "complaint_type": np.random.choice(
        [
            "Late Delivery",
            "Damaged Item",
            "Wrong Item",
            "Missing Item",
            "Customer Service"
        ],
        NUM_COMPLAINTS,
        p=[0.40, 0.20, 0.12, 0.13, 0.15]
    ),

    "severity": np.random.choice(
        [
            "Low",
            "Medium",
            "High",
            "Critical"
        ],
        NUM_COMPLAINTS,
        p=[0.35, 0.40, 0.20, 0.05]
    )
})


# --------------------------------------------------
# 6. INJECT DATA QUALITY PROBLEMS
# --------------------------------------------------

print("Injecting deliberate data-quality issues...")

# Missing customer IDs
missing_customer_rows = np.random.choice(
    orders.index,
    size=2_500,
    replace=False
)

orders.loc[
    missing_customer_rows,
    "customer_id"
] = np.nan


# Negative order values
negative_amount_rows = np.random.choice(
    orders.index,
    size=1_000,
    replace=False
)

orders.loc[
    negative_amount_rows,
    "order_amount"
] *= -1


# Duplicate order records
duplicate_orders = orders.sample(
    n=1_000,
    random_state=24
)

orders = pd.concat(
    [orders, duplicate_orders],
    ignore_index=True
)


# Missing warehouse IDs in deliveries
missing_warehouse_rows = np.random.choice(
    deliveries.index,
    size=750,
    replace=False
)

deliveries.loc[
    missing_warehouse_rows,
    "warehouse_id"
] = np.nan


# --------------------------------------------------
# 7. EXPORT DATA
# --------------------------------------------------

print("Saving datasets...")

orders.to_csv(
    OUTPUT_DIR / "orders.csv",
    index=False
)

deliveries.to_csv(
    OUTPUT_DIR / "deliveries.csv",
    index=False
)

complaints.to_csv(
    OUTPUT_DIR / "complaints.csv",
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nPulseOps data generation complete!")
print("----------------------------------")
print(f"Customers:   {len(customers):,}")
print(f"Warehouses:  {len(warehouses):,}")
print(f"Orders:      {len(orders):,}")
print(f"Deliveries:  {len(deliveries):,}")
print(f"Complaints:  {len(complaints):,}")
print("----------------------------------")
print("Files saved to data/raw/")
