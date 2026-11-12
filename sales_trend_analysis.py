
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
DATASET_FILE = "Sample - Superstore.csv"
OUTPUT_DIR = "sales_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
print("\n" + "=" * 60)
print("       CUSTOMER CHURN / SALES TREND ANALYSIS")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_FILE, encoding="latin1")
    print("\nDataset loaded successfully!")

except FileNotFoundError:
    print("\nERROR: Dataset file not found!")
    print("Keep Sample - Superstore.csv in the same folder as this Python file.")
    raise SystemExit

print("\nOriginal Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\n--- DATA CLEANING ---")

df.columns = df.columns.str.strip()
required_columns = [
    "Order Date", "Customer ID", "Customer Name",
    "Segment", "Category", "Sub-Category",
    "Sales", "Quantity", "Profit"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["Order Date"] = pd.to_datetime(
    df["Order Date"], errors="coerce"
)

if "Ship Date" in df.columns:
    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"], errors="coerce"
    )

for col in ["Sales", "Quantity", "Profit", "Discount"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[
    "Order Date", "Customer ID", "Sales", "Quantity", "Profit"
])

df = df.drop_duplicates()
df = df[
    (df["Sales"] >= 0) &
    (df["Quantity"] > 0)
].copy()

print("Data cleaning completed!")
print("Cleaned Dataset Shape:", df.shape)
print("\n--- FEATURE ENGINEERING ---")
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Month_Name"] = df["Order Date"].dt.strftime("%B")
df["Quarter"] = df["Order Date"].dt.quarter
df["Day"] = df["Order Date"].dt.day
df["Year_Month"] = df["Order Date"].dt.to_period("M").astype(str)
df["Day_Name"] = df["Order Date"].dt.day_name()

df["Revenue"] = df["Sales"]
df["Profit_Margin"] = np.where(
    df["Sales"] != 0,
    (df["Profit"] / df["Sales"]) * 100,
    0
)

if "Discount" in df.columns:
    df["Discount_Percentage"] = df["Discount"] * 100

print("New features created successfully!")

print("\nNew Columns:")
print([
    "Year", "Month", "Month_Name", "Quarter",
    "Day", "Year_Month", "Day_Name",
    "Revenue", "Profit_Margin"
])

print("\n--- OVERALL BUSINESS PERFORMANCE ---")
total_revenue = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
total_customers = df["Customer ID"].nunique()
total_quantity = df["Quantity"].sum()
average_order_value = total_revenue / total_orders if total_orders else 0
print(f"\nTotal Revenue       : ${total_revenue:,.2f}")
print(f"Total Profit        : ${total_profit:,.2f}")
print(f"Total Orders        : {total_orders:,}")
print(f"Total Customers     : {total_customers:,}")
print(f"Total Quantity Sold : {total_quantity:,}")
print(f"Average Order Value : ${average_order_value:,.2f}")

print("\n--- CUSTOMER SEGMENT ANALYSIS ---")
segment_analysis = df.groupby("Segment").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Orders=("Order ID", "nunique"),
    Total_Quantity=("Quantity", "sum"),
    Customer_Count=("Customer ID", "nunique")
).reset_index()
segment_analysis["Sales_Percentage"] = (
    segment_analysis["Total_Sales"] / total_revenue * 100
)

segment_analysis = segment_analysis.sort_values(
    by="Total_Sales", ascending=False
)

print("\nCustomer Segment Performance:")
print(segment_analysis.to_string(index=False))

print("\n--- TOP 10 CUSTOMERS ---")

customer_analysis = df.groupby(
    ["Customer ID", "Customer Name"]
).agg(
    Total_Spending=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Orders=("Order ID", "nunique"),
    Total_Quantity=("Quantity", "sum"),
    Last_Purchase=("Order Date", "max")
).reset_index()

customer_analysis = customer_analysis.sort_values(
    by="Total_Spending", ascending=False
)

print("\nTop 10 Customers:")
print(customer_analysis.head(10).to_string(index=False))

print("\n--- CUSTOMER ACTIVITY ANALYSIS ---")
snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

customer_activity = df.groupby(
    ["Customer ID", "Customer Name"]
).agg(
    Last_Purchase=("Order Date", "max"),
    Frequency=("Order ID", "nunique"),
    Monetary=("Sales", "sum")
).reset_index()

customer_activity["Recency_Days"] = (
    snapshot_date - customer_activity["Last_Purchase"]
).dt.days

customer_activity["Activity_Status"] = np.select(
    [
        customer_activity["Recency_Days"] > 180,
        customer_activity["Recency_Days"] > 90
    ],
    [
        "Inactive in Dataset Window",
        "Needs Attention"
    ],
    default="Recently Active"
)

print("\nCustomer Activity Summary:")
print(
    customer_activity["Activity_Status"]
    .value_counts()
    .to_string()
)

print("\nNote: Activity status is based on the last date in the dataset.")
print("It does not confirm actual customer churn.")
print("\n--- MONTHLY REVENUE ANALYSIS ---")

monthly_sales = df.groupby("Year_Month").agg(
    Monthly_Revenue=("Sales", "sum"),
    Monthly_Profit=("Profit", "sum"),
    Total_Orders=("Order ID", "nunique")
).reset_index()

monthly_sales = monthly_sales.sort_values("Year_Month")
monthly_sales["Revenue_Growth_Percentage"] = (
    monthly_sales["Monthly_Revenue"].pct_change() * 100
)

print("\nMonthly Revenue:")
print(monthly_sales.to_string(index=False))
print("\n--- YEARLY SALES TREND ---")
yearly_sales = df.groupby("Year").agg(
    Yearly_Revenue=("Sales", "sum"),
    Yearly_Profit=("Profit", "sum"),
    Total_Orders=("Order ID", "nunique")
).reset_index()

yearly_sales["Revenue_Growth_Percentage"] = (
    yearly_sales["Yearly_Revenue"].pct_change() * 100
)

print("\nYearly Revenue:")
print(yearly_sales.to_string(index=False))
print("\n--- PIVOT TABLE ANALYSIS ---")
segment_pivot = pd.pivot_table(
    df,
    index="Segment",
    values=["Sales", "Profit", "Quantity"],
    aggfunc="sum",
    fill_value=0
)

print("\nSegment-wise Pivot Table:")
print(segment_pivot)
category_pivot = pd.pivot_table(
    df,
    index="Category",
    columns="Segment",
    values="Sales",
    aggfunc="sum",
    fill_value=0
)

print("\nCategory vs Segment Pivot Table:")
print(category_pivot)
monthly_category_pivot = pd.pivot_table(
    df,
    index="Year_Month",
    columns="Category",
    values="Sales",
    aggfunc="sum",
    fill_value=0
)
print("\nMonthly Category Pivot Table:")
print(monthly_category_pivot)
print("\n--- PRODUCT CATEGORY ANALYSIS ---")
category_analysis = df.groupby("Category").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Quantity=("Quantity", "sum")
).reset_index()
category_analysis = category_analysis.sort_values(
    "Total_Sales", ascending=False
)
print(category_analysis.to_string(index=False))
print("\n--- GENERATING GRAPHS ---")
plt.figure()
plt.plot(
    monthly_sales["Year_Month"],
    monthly_sales["Monthly_Revenue"],
    marker="o",
    linewidth=2
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/monthly_revenue_trend.png")
plt.close()
plt.figure()
sns.barplot(
    data=segment_analysis,
    x="Segment",
    y="Total_Sales"
)
plt.title("Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/customer_segment_sales.png")
plt.close()
plt.figure()
sns.barplot(
    data=category_analysis,
    x="Category",
    y="Total_Sales"
)
plt.title("Category-wise Sales Analysis")
plt.xlabel("Product Category")
plt.ylabel("Total Sales ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/category_wise_sales.png")
plt.close()
top_customers = customer_analysis.head(10)
plt.figure(figsize=(12, 7))
sns.barplot(
    data=top_customers,
    x="Total_Spending",
    y="Customer Name"
)
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Total Spending ($)")
plt.ylabel("Customer Name")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_10_customers.png")
plt.close()
plt.figure()
plt.bar(
    monthly_sales["Year_Month"],
    monthly_sales["Revenue_Growth_Percentage"]
)
plt.axhline(0, linestyle="--")
plt.title("Monthly Revenue Growth (%)")
plt.xlabel("Month")
plt.ylabel("Growth (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/monthly_revenue_growth.png")
plt.close()
plt.figure(figsize=(10, 6))
sns.heatmap(
    category_pivot,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu"
)
plt.title("Sales Heatmap: Category vs Customer Segment")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sales_heatmap.png")
plt.close()
plt.figure()
sns.countplot(
    data=customer_activity,
    x="Activity_Status",
    order=[
        "Recently Active",
        "Needs Attention",
        "Inactive in Dataset Window"
    ]
)
plt.title("Customer Activity Distribution")
plt.xlabel("Customer Activity Status")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/customer_activity.png")
plt.close()
print("All graphs generated successfully!")
print("\n--- BUSINESS RECOMMENDATIONS ---")
recommendations = []
best_segment = segment_analysis.iloc[0]
best_category = category_analysis.iloc[0]
best_customer = customer_analysis.iloc[0]
best_month = monthly_sales.loc[
    monthly_sales["Monthly_Revenue"].idxmax()
]

recommendations.append(
    f"1. {best_segment['Segment']} is the highest revenue "
    f"customer segment. Focus marketing campaigns on this segment."
)

recommendations.append(
    f"2. {best_category['Category']} is the top revenue category. "
    f"Maintain product availability and promote popular products."
)

recommendations.append(
    f"3. {best_customer['Customer Name']} is the highest spending "
    f"customer. Consider loyalty rewards for valuable customers."
)

recommendations.append(
    f"4. {best_month['Year_Month']} recorded the highest monthly "
    f"revenue. Study its sales patterns for future campaigns."
)

needs_attention = (
    customer_activity["Activity_Status"] == "Needs Attention"
).sum()

inactive = (
    customer_activity["Activity_Status"] == "Inactive in Dataset Window"
).sum()

recommendations.append(
    f"5. {needs_attention + inactive} customers have longer gaps "
    f"since their last recorded purchase. Consider personalized "
    f"offers and engagement campaigns."
)

if total_profit < 0:
    recommendations.append(
        "6. Overall profit is negative. Review discounts, product costs "
        "and loss-making categories."
    )
else:
    recommendations.append(
        "6. Monitor profit margins and avoid excessive discounts "
        "to maintain profitability."
    )

for item in recommendations:
    print("\n" + item)

print("\n--- SAVING OUTPUT FILES ---")
df.to_csv(
    f"{OUTPUT_DIR}/cleaned_sales_data.csv",
    index=False
)
segment_analysis.to_csv(
    f"{OUTPUT_DIR}/customer_segment_analysis.csv",
    index=False
)

customer_analysis.to_csv(
    f"{OUTPUT_DIR}/customer_analysis.csv",
    index=False
)

customer_activity.to_csv(
    f"{OUTPUT_DIR}/customer_activity_analysis.csv",
    index=False
)

monthly_sales.to_csv(
    f"{OUTPUT_DIR}/monthly_revenue_analysis.csv",
    index=False
)

yearly_sales.to_csv(
    f"{OUTPUT_DIR}/yearly_revenue_analysis.csv",
    index=False
)

category_analysis.to_csv(
    f"{OUTPUT_DIR}/category_analysis.csv",
    index=False
)

segment_pivot.to_csv(
    f"{OUTPUT_DIR}/segment_pivot_table.csv"
)

category_pivot.to_csv(
    f"{OUTPUT_DIR}/category_segment_pivot_table.csv"
)

monthly_category_pivot.to_csv(
    f"{OUTPUT_DIR}/monthly_category_pivot_table.csv"
)

with open(
    f"{OUTPUT_DIR}/business_recommendations.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("BUSINESS RECOMMENDATIONS REPORT\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Total Revenue: ${total_revenue:,.2f}\n")
    file.write(f"Total Profit: ${total_profit:,.2f}\n")
    file.write(f"Total Orders: {total_orders}\n")
    file.write(f"Total Customers: {total_customers}\n\n")

    for recommendation in recommendations:
        file.write(recommendation + "\n\n")

print("\n" + "=" * 60)
print("              PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\nTotal Revenue: ${total_revenue:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Total Orders: {total_orders}")
print(f"Total Customers: {total_customers}")

print("\nHighest Revenue Segment:", best_segment["Segment"])
print("Top Product Category:", best_category["Category"])
print("Best Revenue Month:", best_month["Year_Month"])

print(f"\nAll results saved in: {OUTPUT_DIR}/")

print("\nGenerated Files:")
print("- Cleaned dataset")
print("- Customer analysis")
print("- Customer activity analysis")
print("- Monthly and yearly revenue analysis")
print("- Pivot tables")
print("- 7 visualization graphs")
print("- Business recommendations report")

print("\nThank you for running the project!")
print("=" * 60)
