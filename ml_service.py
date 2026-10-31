import os
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ShoppingPatternEngine:

    def __init__(self, data_path=os.path.join("data", "online_retail_II.xlsx")):
        self.data_path = data_path
        self.df = None

    def load_and_clean_data(self):
        """Loads all sheets from online_retail_II.xlsx and cleans transaction data."""
        if self.df is not None:
            return self.df

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.data_path}")

        excel_file = pd.ExcelFile(self.data_path)

        # Read and combine all sheets in the Excel workbook
        sheets_data = []
        for sheet_name in excel_file.sheet_names:
            sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name)
            sheets_data.append(sheet_df)

        combined_df = pd.concat(sheets_data, ignore_index=True)

        # 1. Drop rows with missing descriptions
        combined_df = combined_df.dropna(subset=["Description"])

        # 2. Convert Invoice to string
        combined_df["Invoice"] = combined_df["Invoice"].astype(str)

        # 3. Filter out cancelled transactions (Invoices starting with 'C')
        combined_df = combined_df[~combined_df["Invoice"].str.startswith("C")]

        # 4. Keep positive quantities and prices
        combined_df = combined_df[
            (combined_df["Quantity"] > 0) & (combined_df["Price"] > 0)
        ]

        # 5. Clean product descriptions and calculate line spend
        combined_df["Description"] = combined_df["Description"].str.strip()
        combined_df["TotalAmount"] = (
            combined_df["Quantity"] * combined_df["Price"]
        )

        self.df = combined_df
        return self.df

    def get_dashboard_summary(self):
        """Returns top-level metric KPIs for the dashboard UI."""
        df = self.load_and_clean_data()

        total_revenue = float(df["TotalAmount"].sum())
        total_orders = int(df["Invoice"].nunique())
        total_products = int(df["Description"].nunique())

        valid_customers = df.dropna(subset=["Customer ID"])
        total_customers = int(valid_customers["Customer ID"].nunique())

        return {
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "total_products": total_products,
            "total_customers": total_customers,
        }

    def get_association_rules(
        self, min_support=0.02, min_confidence=0.3, min_lift=1.0
    ):
        """Generates market basket association rules using FP-Growth algorithm."""
        df = self.load_and_clean_data()

        # Build basket pivot table: Invoice x Description
        basket = (
            df.groupby(["Invoice", "Description"])["Quantity"]
            .sum()
            .unstack()
            .reset_index()
            .fillna(0)
            .set_index("Invoice")
        )

        # Encode quantities to binary 0 or 1
        basket_binary = basket.map(lambda x: 1 if x > 0 else 0)

        # Run FP-Growth
        frequent_itemsets = fpgrowth(
            basket_binary, min_support=min_support, use_colnames=True
        )

        if frequent_itemsets.empty:
            return []

        # Generate Association Rules
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        rules = rules[rules["lift"] >= min_lift].sort_values(
            by="lift", ascending=False
        )

        # Format rules list for JSON API output
        results = []
        for _, row in rules.iterrows():
            results.append(
                {
                    "antecedent": list(row["antecedents"])[0],
                    "consequent": list(row["consequents"])[0],
                    "support": round(float(row["support"]), 4),
                    "confidence": round(float(row["confidence"]), 4),
                    "lift": round(float(row["lift"]), 4),
                }
            )
        return results

    def get_customer_clusters(self, n_clusters=4):
        """Aggregates RFM metrics and performs K-Means customer segmentation."""
        df = self.load_and_clean_data()

        df_cust = df.dropna(subset=["Customer ID"]).copy()
        df_cust["Customer ID"] = df_cust["Customer ID"].astype(int).astype(str)

        max_date = df_cust["InvoiceDate"].max() + pd.Timedelta(days=1)

        # Aggregate Recency, Frequency, and Monetary (RFM)
        rfm = (
            df_cust.groupby("Customer ID")
            .agg(
                {
                    "InvoiceDate": lambda x: (max_date - x.max()).days,
                    "Invoice": "nunique",
                    "TotalAmount": "sum",
                }
            )
            .rename(
                columns={
                    "InvoiceDate": "Recency",
                    "Invoice": "Frequency",
                    "TotalAmount": "Monetary",
                }
            )
        )

        # Feature scaling using Z-Score Standardization
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        # K-Means Model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

        # Map cluster labels
        cluster_labels = {
            0: "Loyal Customers",
            1: "At Risk / Inactive",
            2: "High Spenders",
            3: "Occasional Buyers",
        }
        rfm["Segment"] = rfm["Cluster"].map(
            lambda c: cluster_labels.get(c, f"Cluster {c}")
        )

        return rfm.reset_index().to_dict(orient="records")