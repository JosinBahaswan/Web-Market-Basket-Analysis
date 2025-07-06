# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Additional imports for new features
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import json
import base64
from io import BytesIO
import zipfile
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Analisis Keranjang Belanja",
    page_icon="🛒",
    layout="wide"
)

# Title and description
st.title("Analisis Keranjang Belanja - Algoritma Apriori")
st.markdown("""
Aplikasi ini melakukan analisis keranjang belanja menggunakan algoritma Apriori untuk menemukan 
asosiasi antar produk yang sering dibeli bersamaan.
""")

# File uploader with default to Data_Transaksi.csv
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

# Function to preprocess data
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    try:
        if uploaded_file is not None:
            # Try to read with skiprows=2 first (to skip the kaggle link and empty line)
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', skiprows=2)
        else:
            # Try to read with skiprows=2 first (to skip the kaggle link and empty line)
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', skiprows=2)
        
        # Check if we have the correct headers
        if 'TransactionID' not in df.columns:
            # If not, try different approaches
            if uploaded_file is not None:
                uploaded_file.seek(0)
                # Read the first few lines to analyze
                header_lines = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', nrows=5)
                
                # Find the row with TransactionID
                for i, row in header_lines.iterrows():
                    if any('TransactionID' in str(val) for val in row.values):
                        header_row = i
                        break
                
                # Re-read with the correct header row
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', skiprows=header_row)
            else:
                # Read the first few lines to analyze
                header_lines = pd.read_csv(file_path, sep=';', encoding='utf-8', nrows=5)
                
                # Find the row with TransactionID
                for i, row in header_lines.iterrows():
                    if any('TransactionID' in str(val) for val in row.values):
                        header_row = i
                        break
                
                # Re-read with the correct header row
                df = pd.read_csv(file_path, sep=';', encoding='utf-8', skiprows=header_row)
        
        # Display info for debugging
        st.write(f"Kolom yang ditemukan: {df.columns.tolist()}")
        st.write(f"Jumlah baris: {len(df)}")
        
        # Show a sample of the data
        st.write("Contoh data yang dimuat:")
        st.write(df.head(3))
        
        return df
    except Exception as e:
        st.error(f"Error saat memuat data: {e}")
        return None

# Function to prepare transactions
@st.cache_data
def prepare_transactions(df):
    # Convert the Products column to a list of lists
    if 'Products' in df.columns:
        # Show raw product data for debugging
        st.write("Data produk mentah (3 baris pertama):")
        for i in range(min(3, len(df))):
            st.write(f"Baris {i+1}: {df['Products'].iloc[i]}")
        
        # Check if the data is already in the correct format
        sample = df['Products'].iloc[0] if not df.empty else ""
        
        if isinstance(sample, str):
            # Enhanced data cleaning and processing
            transactions = []
            for products in df['Products']:
                if isinstance(products, str) and products.strip():
                    # Remove quotes and extra whitespace
                    products = products.strip().strip('"').strip("'")
                    
                    # Handle different separators
                    if ',' in products:
                        # Split by comma
                        items = [item.strip() for item in products.split(',')]
                    elif ';' in products:
                        # Split by semicolon
                        items = [item.strip() for item in products.split(';')]
                    elif '|' in products:
                        # Split by pipe
                        items = [item.strip() for item in products.split('|')]
                    else:
                        # Single item or space-separated
                        items = [products.strip()]
                    
                    # Filter out empty items and clean up
                    items = [item for item in items if item and len(item) > 1]  # Remove single characters
                    
                    # Additional cleaning for common issues
                    cleaned_items = []
                    for item in items:
                        # Remove extra quotes and parentheses
                        item = item.strip().strip('"').strip("'").strip('()').strip('[]')
                        if item and len(item) > 1:  # Only keep items with more than 1 character
                            cleaned_items.append(item)
                    
                    transactions.append(cleaned_items)
                else:
                    transactions.append([])
        else:
            st.error("Kolom Products tidak berisi data string")
            return []
        
        # Debug info
        st.write(f"Jumlah transaksi: {len(transactions)}")
        st.write("Contoh transaksi yang diproses:")
        st.write(transactions[:3])
        
        # Additional validation
        if transactions:
            # Check if we have valid transactions
            valid_transactions = [t for t in transactions if len(t) > 0]
            if len(valid_transactions) == 0:
                st.error("Tidak ada transaksi valid yang ditemukan setelah preprocessing")
                return []
            
            # Show unique products for debugging
            all_products = set()
            for transaction in valid_transactions:
                all_products.update(transaction)
            
            st.write(f"Jumlah produk unik: {len(all_products)}")
            st.write("Contoh produk unik (10 pertama):")
            st.write(list(all_products)[:10])
        
        return transactions
    else:
        st.error(f"Kolom 'Products' tidak ditemukan. Kolom yang tersedia: {df.columns.tolist()}")
        return []

# Function to create one-hot encoded DataFrame
@st.cache_data
def create_one_hot_df(transactions):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Debug info
    st.write(f"Bentuk data one-hot encoded: {df_encoded.shape}")
    st.write(f"Jumlah produk unik: {len(df_encoded.columns)}")
    
    return df_encoded

# Function to apply Apriori algorithm
@st.cache_data
# Tambahkan display_format agar cache invalid saat mode berubah

def apply_apriori(df_encoded, min_support, display_format=None):
    try:
        # Ensure min_support is not too high
        if min_support > 0.5:
            st.warning("Nilai support terlalu tinggi. Menurunkan ke 0.01.")
            min_support = 0.01
        
        # Check if df_encoded is valid
        if df_encoded.empty:
            st.error("Dataframe yang di-encode kosong. Tidak dapat menerapkan algoritma Apriori.")
            return pd.DataFrame(columns=['support', 'itemsets'])
        
        # Display sample of encoded data
        st.write("Contoh data yang di-encode (3 baris pertama, 5 kolom pertama):")
        sample_cols = min(5, len(df_encoded.columns))
        st.write(df_encoded.iloc[:3, :sample_cols])
        
        # Apply apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # Debug info
        if len(frequent_itemsets) == 0:
            st.warning(f"Tidak ditemukan itemset yang sering dengan threshold support {min_support}. Coba turunkan nilai support.")
        else:
            st.success(f"Ditemukan {len(frequent_itemsets)} itemset yang sering.")
            
        return frequent_itemsets
    except Exception as e:
        st.error(f"Error dalam algoritma Apriori: {e}")
        return pd.DataFrame(columns=['support', 'itemsets'])

# Function to generate association rules
@st.cache_data
# Tambahkan display_format agar cache invalid saat mode berubah

def generate_rules(frequent_itemsets, min_threshold, display_format=None):
    try:
        if len(frequent_itemsets) == 0:
            st.warning("Tidak ada itemset yang sering tersedia untuk menghasilkan aturan.")
            return pd.DataFrame()
        
        # Check if the frequent_itemsets DataFrame has the required format
        if 'support' not in frequent_itemsets.columns or 'itemsets' not in frequent_itemsets.columns:
            st.error("Dataframe itemset yang sering tidak memiliki kolom yang diperlukan (support, itemsets).")
            return pd.DataFrame()
        
        # Ensure 'itemsets' column contains frozenset objects
        if not all(isinstance(x, frozenset) for x in frequent_itemsets['itemsets']):
            st.error("Kolom 'itemsets' tidak berisi objek frozenset. Silakan periksa preprocessing data Anda.")
            st.write(frequent_itemsets.head(3))
            return pd.DataFrame()
        
        # Display sample of frequent itemsets
        st.write("Contoh itemset yang sering:")
        st.write(frequent_itemsets.head(3))
        
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
        
        # Debug info
        if len(rules) == 0:
            st.warning(f"Tidak ditemukan aturan dengan threshold confidence {min_threshold}. Coba turunkan nilai confidence.")
        else:
            st.success(f"Menghasilkan {len(rules)} aturan asosiasi.")
            
        return rules
    except Exception as e:
        st.error(f"Error menghasilkan aturan asosiasi: {str(e)}\n\nKemungkinan besar data frequent_itemsets tidak valid. Pastikan parameter support tidak terlalu tinggi dan data transaksi sudah benar.")
        st.info("Tips: Cek apakah data transaksi sudah benar dan parameter support tidak terlalu tinggi.")
        return pd.DataFrame()

# Function to visualize frequent items
def plot_frequent_items(transactions):
    # Count item frequencies
    item_counts = Counter([item for sublist in transactions for item in sublist])
    
    # Convert to DataFrame
    item_df = pd.DataFrame({
        'Item': list(item_counts.keys()),
        'Frequency': list(item_counts.values())
    }).sort_values('Frequency', ascending=False)
    
    # Plot top 15 items
    fig = px.bar(
        item_df.head(15), 
        x='Item', 
        y='Frequency',
        title='15 Item Paling Sering Dibeli',
        color='Frequency',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    return item_df

# Function to visualize association rules network
def plot_association_network(rules, min_lift=1, max_rules=20):
    if len(rules) == 0:
        st.warning("Tidak ada aturan untuk divisualisasikan. Coba turunkan nilai minimum confidence atau lift.")
        return
    
    # Filter rules by lift
    filtered_rules = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False).head(max_rules)
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add edges
    for _, row in filtered_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        for antecedent in antecedents:
            for consequent in consequents:
                G.add_edge(antecedent, consequent, 
                           weight=row['lift'],
                           confidence=row['confidence'],
                           support=row['support'])
    
    # Create positions for nodes
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', alpha=0.8)
    
    # Draw edges with varying thickness based on lift
    edges = G.edges()
    weights = [G[u][v]['weight'] * 0.5 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', arrows=True, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(plt)

# Function to plot heatmap of rules
def plot_rules_heatmap(rules, metric='lift', max_rules=15):
    if len(rules) == 0:
        st.warning("Tidak ada aturan untuk divisualisasikan. Coba turunkan nilai minimum confidence atau lift.")
        return
    
    # Take top N rules
    top_rules = rules.sort_values(metric, ascending=False).head(max_rules)
    
    # Create labels for the heatmap
    labels = [f"{', '.join(list(x))} → {', '.join(list(y))}" for x, y in zip(top_rules['antecedents'], top_rules['consequents'])]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=top_rules[['support', 'confidence', 'lift']].values,
        x=['Support', 'Confidence', 'Lift'],
        y=labels,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'Top {max_rules} Aturan Asosiasi berdasarkan {metric.capitalize()}',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Function to create scatter plot of rules
def plot_rules_scatter(rules):
    if len(rules) == 0:
        st.warning("Tidak ada aturan untuk divisualisasikan. Coba turunkan nilai minimum confidence atau lift.")
        return
    
    # Create scatter plot
    fig = px.scatter(
        rules, x='support', y='confidence', 
        size='lift', color='lift',
        hover_name=[f"{', '.join(list(x))} → {', '.join(list(y))}" for x, y in zip(rules['antecedents'], rules['consequents'])],
        log_x=True, size_max=20,
        color_continuous_scale='Viridis',
        title='Aturan Asosiasi - Support vs Confidence'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===== NEW FEATURES =====

# Function for temporal analysis
@st.cache_data
def analyze_temporal_patterns(df, transactions):
    """Analyze temporal patterns in transaction data"""
    try:
        # Find date/timestamp column
        date_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'datetime', 'created', 'updated']):
                date_columns.append(col)
        
        if date_columns:
            # Use the first found date column
            date_col = date_columns[0]
            st.info(f"📅 Menggunakan kolom '{date_col}' untuk analisis temporal")
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Remove rows with invalid dates
            valid_dates = df[date_col].notna()
            if valid_dates.sum() == 0:
                st.warning(f"⚠️ Tidak ada tanggal valid dalam kolom '{date_col}'")
                return {'has_temporal': False}
            
            df_valid = df[valid_dates].copy()
            
            # Create temporal features
            df_valid['Hour'] = df_valid[date_col].dt.hour
            df_valid['DayOfWeek'] = df_valid[date_col].dt.day_name()
            df_valid['Month'] = df_valid[date_col].dt.month
            df_valid['DayOfMonth'] = df_valid[date_col].dt.day
            df_valid['Year'] = df_valid[date_col].dt.year
            
            # Hourly analysis
            hourly_counts = df_valid['Hour'].value_counts().sort_index()
            
            # Daily analysis
            daily_counts = df_valid['DayOfWeek'].value_counts()
            daily_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_counts = daily_counts.reindex(daily_order)
            
            # Monthly analysis
            monthly_counts = df_valid['Month'].value_counts().sort_index()
            
            # Year analysis
            year_counts = df_valid['Year'].value_counts().sort_index()
            
            # Date range info
            date_range = {
                'start_date': df_valid[date_col].min(),
                'end_date': df_valid[date_col].max(),
                'total_days': (df_valid[date_col].max() - df_valid[date_col].min()).days
            }
            
            return {
                'hourly': hourly_counts,
                'daily': daily_counts,
                'monthly': monthly_counts,
                'yearly': year_counts,
                'date_range': date_range,
                'date_column': date_col,
                'has_temporal': True
            }
        else:
            st.info("ℹ️ Tidak ditemukan kolom tanggal/timestamp dalam data")
            return {'has_temporal': False}
    except Exception as e:
        st.error(f"Error dalam analisis temporal: {e}")
        return {'has_temporal': False}

# Function for Product Clustering
@st.cache_data
def perform_product_clustering(df_encoded, transactions):
    """Perform clustering analysis on products"""
    try:
        # Calculate product features
        product_features = []
        product_names = list(df_encoded.columns)
        
        for product in product_names:
            # Frequency
            frequency = df_encoded[product].sum()
            
            # Average basket size when product is present
            baskets_with_product = df_encoded[df_encoded[product] == 1]
            avg_basket_size = baskets_with_product.sum(axis=1).mean() if len(baskets_with_product) > 0 else 0
            
            # Support
            support = frequency / len(df_encoded)
            
            product_features.append({
                'product': product,
                'frequency': frequency,
                'avg_basket_size': avg_basket_size,
                'support': support
            })
        
        features_df = pd.DataFrame(product_features)
        
        if len(features_df) > 3:  # Need at least 3 products for clustering
            # Prepare features for clustering
            X = features_df[['frequency', 'avg_basket_size', 'support']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform K-means clustering
            n_clusters = min(5, len(features_df) // 2)  # Max 5 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            features_df['cluster'] = kmeans.fit_predict(X_scaled)
            
            return features_df
        else:
            return None
    except Exception as e:
        st.error(f"Error dalam product clustering: {e}")
        return None

# Function for Basket Profitability Analysis
@st.cache_data
def analyze_basket_profitability(transactions, rules):
    """Analyze profitability of different basket combinations"""
    try:
        if len(rules) == 0:
            return None
        
        # Simulate product prices (in real scenario, this would come from product database)
        product_prices = {}
        all_products = set()
        for transaction in transactions:
            all_products.update(transaction)
        
        # Generate random prices for demonstration
        np.random.seed(42)
        for product in all_products:
            product_prices[product] = np.random.uniform(10, 100)
        
        # Calculate basket profitability for top rules
        top_rules = rules.head(10)
        profitability_data = []
        
        for _, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Calculate basket value
            basket_items = antecedents + consequents
            basket_value = sum(product_prices.get(item, 0) for item in basket_items)
            
            # Calculate profit margin (assume 30% margin)
            profit_margin = 0.3
            basket_profit = basket_value * profit_margin
            
            profitability_data.append({
                'antecedents': ', '.join(antecedents),
                'consequents': ', '.join(consequents),
                'basket_value': basket_value,
                'basket_profit': basket_profit,
                'lift': rule['lift'],
                'confidence': rule['confidence']
            })
        
        return pd.DataFrame(profitability_data)
    except Exception as e:
        st.error(f"Error dalam basket profitability analysis: {e}")
        return None

# Function for Anomaly Detection
@st.cache_data
def detect_anomalies(df_encoded, transactions):
    """Detect anomalous transaction patterns"""
    try:
        # Calculate transaction sizes
        transaction_sizes = [len(transaction) for transaction in transactions]
        
        # Calculate statistics
        mean_size = np.mean(transaction_sizes)
        std_size = np.std(transaction_sizes)
        
        # Define anomalies (transactions with size > mean + 2*std)
        threshold = mean_size + 2 * std_size
        anomalies = [i for i, size in enumerate(transaction_sizes) if size > threshold]
        
        # Calculate product frequency anomalies
        product_frequencies = df_encoded.sum()
        mean_freq = product_frequencies.mean()
        std_freq = product_frequencies.std()
        freq_threshold = mean_freq + 2 * std_freq
        
        freq_anomalies = product_frequencies[product_frequencies > freq_threshold]
        
        return {
            'size_anomalies': anomalies,
            'frequency_anomalies': freq_anomalies,
            'mean_size': mean_size,
            'std_size': std_size,
            'threshold': threshold
        }
    except Exception as e:
        st.error(f"Error dalam anomaly detection: {e}")
        return None

# Function for Export Reports
def export_reports(rules, frequent_itemsets, clustering_results=None):
    """Export analysis results to various formats"""
    try:
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            files_to_zip = []
            
            # Export rules to CSV
            if not rules.empty:
                rules_csv = os.path.join(temp_dir, 'association_rules.csv')
                rules_export = rules.copy()
                rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
                rules_export.to_csv(rules_csv, index=False)
                files_to_zip.append(rules_csv)
            
            # Export frequent itemsets to CSV
            if not frequent_itemsets.empty:
                itemsets_csv = os.path.join(temp_dir, 'frequent_itemsets.csv')
                itemsets_export = frequent_itemsets.copy()
                itemsets_export['itemsets'] = itemsets_export['itemsets'].apply(lambda x: ', '.join(list(x)))
                itemsets_export.to_csv(itemsets_csv, index=False)
                files_to_zip.append(itemsets_csv)
            
            # Export clustering results
            if clustering_results is not None:
                clustering_csv = os.path.join(temp_dir, 'product_clustering.csv')
                clustering_results.to_csv(clustering_csv, index=False)
                files_to_zip.append(clustering_csv)
            
            # Create summary report
            summary_report = os.path.join(temp_dir, 'summary_report.txt')
            with open(summary_report, 'w', encoding='utf-8') as f:
                f.write("ANALISIS KERANJANG BELANJA - LAPORAN SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Aturan Asosiasi: {len(rules)}\n")
                f.write(f"Total Itemset Sering: {len(frequent_itemsets)}\n")
                if clustering_results is not None:
                    f.write(f"Total Produk Clustering: {len(clustering_results)}\n")
                f.write(f"\nTanggal Export: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            files_to_zip.append(summary_report)
            
            # Create ZIP file
            zip_path = os.path.join(temp_dir, 'analysis_reports.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in files_to_zip:
                    zipf.write(file, os.path.basename(file))
            
            # Read ZIP file for download
            with open(zip_path, 'rb') as f:
                zip_data = f.read()
            
            return zip_data
            
    except Exception as e:
        st.error(f"Error dalam export reports: {e}")
        return None

# Function for Advanced Visualizations
def create_advanced_visualizations(temporal_data, clustering_results, anomalies):
    """Create advanced visualizations for the analysis"""
    
    # Temporal Analysis Visualization
    if temporal_data.get('has_temporal', False):
        st.subheader("📊 Analisis Temporal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly heatmap
            if 'hourly' in temporal_data:
                fig_hourly = px.bar(
                    x=temporal_data['hourly'].index,
                    y=temporal_data['hourly'].values,
                    title='Distribusi Transaksi per Jam',
                    labels={'x': 'Jam', 'y': 'Jumlah Transaksi'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Daily pattern
            if 'daily' in temporal_data:
                fig_daily = px.bar(
                    x=temporal_data['daily'].index,
                    y=temporal_data['daily'].values,
                    title='Distribusi Transaksi per Hari',
                    labels={'x': 'Hari', 'y': 'Jumlah Transaksi'}
                )
                st.plotly_chart(fig_daily, use_container_width=True)
    
    # Clustering Visualization
    if clustering_results is not None:
        st.subheader("🏷️ Analisis Clustering Produk")
        
        # PCA for visualization
        features = clustering_results[['frequency', 'avg_basket_size', 'support']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        clustering_results['PCA1'] = features_pca[:, 0]
        clustering_results['PCA2'] = features_pca[:, 1]
        
        fig_cluster = px.scatter(
            clustering_results,
            x='PCA1',
            y='PCA2',
            color='cluster',
            hover_name='product',
            title='Clustering Produk (PCA Visualization)',
            labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Anomaly Detection Visualization
    if anomalies is not None:
        st.subheader("🚨 Deteksi Anomali")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction size distribution
            transaction_sizes = [len(transaction) for transaction in transactions]
            fig_size_dist = px.histogram(
                x=transaction_sizes,
                title='Distribusi Ukuran Transaksi',
                labels={'x': 'Ukuran Transaksi', 'y': 'Frekuensi'}
            )
            fig_size_dist.add_vline(x=anomalies['threshold'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_size_dist, use_container_width=True)
        
        with col2:
            # Anomaly summary
            st.metric("Total Anomali Ukuran", len(anomalies['size_anomalies']))
            st.metric("Total Anomali Frekuensi", len(anomalies['frequency_anomalies']))
            st.metric("Threshold Anomali", f"{anomalies['threshold']:.1f}")



# Main application logic
try:
    # Load data
    if uploaded_file is not None:
        df = load_data(uploaded_file=uploaded_file)
    else:
        df = load_data(file_path='Data_Transaksi.csv')
    
    if df is None or df.empty:
        st.error("Gagal memuat data atau data kosong.")
        st.stop()
    
    # Display raw data
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df)
    
    # Debug: Show column information
    st.subheader("🔍 Informasi Kolom Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kolom yang tersedia:**")
        for i, col in enumerate(df.columns):
            st.write(f"{i+1}. `{col}`")
    
    with col2:
        st.write("**Tipe data kolom:**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_value = str(df[col].iloc[0]) if not df.empty else "N/A"
            st.write(f"`{col}`: {dtype} (contoh: {sample_value[:50]})")
    
    # Check for date columns
    date_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'datetime', 'created', 'updated']):
            date_columns.append(col)
    
    if date_columns:
        st.success(f"✅ Ditemukan {len(date_columns)} kolom tanggal: {', '.join(date_columns)}")
    else:
        st.warning("⚠️ Tidak ditemukan kolom tanggal dalam data")
    
    # Prepare transactions
    transactions = prepare_transactions(df)
    
    if not transactions:
        st.error("Tidak ada transaksi untuk dianalisis.")
        st.stop()
    
    # Display transaction sample
    with st.expander("Lihat Contoh Transaksi"):
        st.write(transactions[:5])
    
    # Sidebar for parameter display format
    st.sidebar.header("Format Tampilan Parameter")
    display_format = st.sidebar.radio(
        "Format Tampilan Parameter:",
        ["Persentase (%)", "Desimal (0-1)"],
        index=0  # Default ke persentase
    )
    
    # Sidebar for parameter explanation
    with st.sidebar.expander("Penjelasan Parameter"):
        st.markdown("""
        ### Support (Dukungan)
        - **Definisi**: Persentase transaksi yang mengandung itemset tertentu
        - **Contoh**: Support 5% untuk "Roti" berarti 5% dari seluruh transaksi mengandung item "Roti"
        - **Rumus**: Support(A) = (Jumlah transaksi mengandung A) / (Total transaksi)
        - **Pengaruh**:
            - Nilai rendah (0.1% - 1%): Menemukan lebih banyak pola termasuk yang jarang terjadi
            - Nilai tinggi (5% - 10%): Hanya menemukan pola yang sangat umum
        
        ### Confidence (Kepercayaan)
        - **Definisi**: Persentase transaksi yang mengandung item B jika mengandung item A
        - **Contoh**: Confidence 60% untuk aturan "Roti → Selai" berarti 60% transaksi yang berisi "Roti" juga berisi "Selai"
        - **Rumus**: Confidence(A→B) = Support(A,B) / Support(A)
        - **Pengaruh**:
            - Nilai rendah (5% - 20%): Menghasilkan lebih banyak aturan termasuk yang kurang kuat
            - Nilai tinggi (50% - 80%): Hanya menghasilkan aturan yang sangat kuat
        
        ### Lift (Pengangkatan)
        - **Definisi**: Rasio kemunculan bersama dibandingkan dengan kemunculan independen
        - **Rumus**: Lift(A→B) = Confidence(A→B) / Support(B)
        - **Interpretasi**:
            - Lift > 1: Item cenderung dibeli bersama (asosiasi positif)
            - Lift = 1: Tidak ada asosiasi
            - Lift < 1: Item cenderung tidak dibeli bersama (asosiasi negatif)
        """)
    
    # Add preset parameter options
    st.sidebar.subheader("Preset Parameter:")
    preset = st.sidebar.selectbox(
        "Pilih Preset:",
        ["Custom", "Dataset Kecil (<1000 transaksi)", "Dataset Besar (>1000 transaksi)"]
    )
    
    # Set parameters based on display format and preset
    if display_format == "Persentase (%)":
        if preset == "Dataset Kecil (<1000 transaksi)":
            min_support_pct = 1.0  # 1%
            min_confidence_pct = 20.0  # 20%
        elif preset == "Dataset Besar (>1000 transaksi)":
            min_support_pct = 0.5  # 0.5%
            min_confidence_pct = 10.0  # 10%
        else:  # Custom
            min_support_pct = st.sidebar.slider(
                "Minimum Support (%)",
                min_value=0.1,
                max_value=20.0,
                value=1.0,
                step=0.1,
                help="Persentase minimum transaksi yang harus mengandung itemset"
            )
            
            min_confidence_pct = st.sidebar.slider(
                "Minimum Confidence (%)",
                min_value=5.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
                help="Persentase minimum kepercayaan untuk aturan asosiasi"
            )
        
        # Convert percentage to proportion
        min_support = min_support_pct / 100.0
        min_confidence = min_confidence_pct / 100.0
    else:  # Desimal (0-1)
        if preset == "Dataset Kecil (<1000 transaksi)":
            min_support = 0.01  # 1%
            min_confidence = 0.2  # 20%
        elif preset == "Dataset Besar (>1000 transaksi)":
            min_support = 0.005  # 0.5%
            min_confidence = 0.1  # 10%
        else:  # Custom
            min_support = st.sidebar.slider(
                "Minimum Support (0-1)",
                min_value=0.001,
                max_value=0.2,
                value=0.01,
                step=0.001,
                help="Proporsi minimum transaksi yang harus mengandung itemset"
            )
            
            min_confidence = st.sidebar.slider(
                "Minimum Confidence (0-1)",
                min_value=0.05,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Proporsi minimum kepercayaan untuk aturan asosiasi"
            )
    
    min_lift = st.sidebar.slider(
        "Minimum Lift",
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Nilai minimum lift untuk aturan asosiasi (>1 menunjukkan asosiasi positif)"
    )
    
    # Display parameter values
    st.sidebar.subheader("Parameter yang Digunakan:")
    if display_format == "Persentase (%)":
        st.sidebar.write(f"- Support: {min_support*100:.2f}%")
        st.sidebar.write(f"- Confidence: {min_confidence*100:.2f}%")
    else:
        st.sidebar.write(f"- Support: {min_support:.4f}")
        st.sidebar.write(f"- Confidence: {min_confidence:.4f}")
    st.sidebar.write(f"- Lift: {min_lift:.2f}")
    
    # Create one-hot encoded DataFrame
    df_encoded = create_one_hot_df(transactions)
    
    # Apply Apriori algorithm
    with st.spinner("Menerapkan algoritma Apriori..."):
        frequent_itemsets = apply_apriori(df_encoded, min_support, display_format)
    
    # Display frequent itemsets
    st.header("Itemset yang Sering")
    st.write(f"Jumlah itemset yang sering: {len(frequent_itemsets)}")
    
    with st.expander("Lihat Itemset yang Sering"):
        if not frequent_itemsets.empty:
            # Format support as percentage jika perlu, tapi JANGAN ubah kolom 'itemsets' pada DataFrame utama
            if display_format == "Persentase (%)":
                frequent_itemsets_display = frequent_itemsets.copy()
                frequent_itemsets_display['support'] = frequent_itemsets_display['support'] * 100
                frequent_itemsets_display['itemsets'] = frequent_itemsets_display['itemsets'].apply(lambda x: ', '.join(list(x)))
                frequent_itemsets_display.rename(columns={'support': 'support (%)'}, inplace=True)
                st.dataframe(frequent_itemsets_display)
            else:
                # Untuk tampilan desimal, tampilkan apa adanya, JANGAN ubah kolom 'itemsets'
                st.dataframe(frequent_itemsets)
        else:
            st.warning("Tidak ditemukan itemset yang sering. Coba turunkan threshold support.")
    
    # Generate association rules
    with st.spinner("Menghasilkan aturan asosiasi..."):
        if not frequent_itemsets.empty:
            rules = generate_rules(frequent_itemsets, min_confidence, display_format)
            
            # Filter by lift
            if not rules.empty:
                rules = rules[rules['lift'] >= min_lift]
        else:
            rules = pd.DataFrame()
    
    # Display association rules
    st.header("Aturan Asosiasi")
    st.write(f"Jumlah aturan: {len(rules)}")
    
    with st.expander("Lihat Aturan Asosiasi"):
        if not rules.empty:
            # Format the antecedents and consequents for better readability
            rules_display = rules.copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Format support and confidence as percentage if needed
            if display_format == "Persentase (%)":
                rules_display['support'] = rules_display['support'] * 100
                rules_display['confidence'] = rules_display['confidence'] * 100
                rules_display.rename(columns={
                    'support': 'support (%)',
                    'confidence': 'confidence (%)'
                }, inplace=True)
            
            st.dataframe(rules_display)
        else:
            st.warning("Tidak ditemukan aturan asosiasi. Coba sesuaikan parameter.")
    
    # Visualizations
    if not frequent_itemsets.empty:
        st.header("Visualisasi")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Item Paling Sering", "Jaringan Asosiasi", "Heatmap Aturan", "Scatter Plot Aturan",
            "Analisis Lanjutan", "Export Laporan"
        ])
        
        with tab1:
            st.subheader("Item Paling Sering")
            item_df = plot_frequent_items(transactions)
        
        if not rules.empty:
            with tab2:
                st.subheader("Jaringan Asosiasi")
                plot_association_network(rules, min_lift=min_lift)
            
            with tab3:
                st.subheader("Heatmap Aturan")
                plot_rules_heatmap(rules)
            
            with tab4:
                st.subheader("Scatter Plot Aturan")
                plot_rules_scatter(rules)
            
            with tab5:
                st.subheader("📊 Analisis Lanjutan")
                
                # Advanced Analysis Section
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔍 Analisis Temporal")
                    temporal_data = analyze_temporal_patterns(df, transactions)
                    if temporal_data.get('has_temporal', False):
                        st.success("✅ Data temporal tersedia")
                        
                        # Show date range information
                        if 'date_range' in temporal_data:
                            date_range = temporal_data['date_range']
                            st.write("**Rentang Data:**")
                            st.write(f"- Mulai: {date_range['start_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write(f"- Selesai: {date_range['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write(f"- Total hari: {date_range['total_days']} hari")
                        
                        if 'hourly' in temporal_data:
                            st.write("**Jam Puncak Transaksi:**")
                            peak_hour = temporal_data['hourly'].idxmax()
                            st.metric("Jam Puncak", f"{peak_hour}:00")
                        
                        if 'daily' in temporal_data:
                            st.write("**Hari Puncak Transaksi:**")
                            peak_day = temporal_data['daily'].idxmax()
                            st.metric("Hari Puncak", peak_day)
                    else:
                        st.info("ℹ️ Data temporal tidak tersedia (kolom 'Date' tidak ditemukan)")
                
                with col2:
                    st.subheader("🚨 Deteksi Anomali")
                    anomalies = detect_anomalies(df_encoded, transactions)
                    if anomalies is not None:
                        st.success("✅ Anomaly detection berhasil")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Anomali Ukuran", len(anomalies['size_anomalies']))
                        with col2:
                            st.metric("Anomali Frekuensi", len(anomalies['frequency_anomalies']))
                        with col3:
                            st.metric("Threshold", f"{anomalies['threshold']:.1f}")
                        
                        if len(anomalies['size_anomalies']) > 0:
                            st.warning(f"⚠️ Ditemukan {len(anomalies['size_anomalies'])} transaksi dengan ukuran anomali")
                        
                        if len(anomalies['frequency_anomalies']) > 0:
                            st.warning(f"⚠️ Ditemukan {len(anomalies['frequency_anomalies'])} produk dengan frekuensi anomali")
                    else:
                        st.info("ℹ️ Anomaly detection tidak tersedia")
            
            with tab6:
                st.subheader("📊 Export Laporan")
                
                st.write("**Export hasil analisis dalam berbagai format:**")
                
                # Export options
                export_options = st.multiselect(
                    "Pilih data untuk di-export:",
                    ["Aturan Asosiasi", "Itemset Sering", "Clustering Produk", "Laporan Summary"],
                    default=["Aturan Asosiasi", "Itemset Sering", "Laporan Summary"]
                )
                
                if st.button("📥 Download Laporan (ZIP)", type="primary"):
                    with st.spinner("Menyiapkan laporan..."):
                        # Prepare data for export
                        export_clustering = clustering_results if "Clustering Produk" in export_options else None
                        
                        # Generate export
                        zip_data = export_reports(rules, frequent_itemsets, export_clustering)
                        
                        if zip_data:
                            # Create download button
                            st.download_button(
                                label="💾 Download ZIP File",
                                data=zip_data,
                                file_name=f"analisis_keranjang_belanja_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
                            st.success("✅ File ZIP berhasil dibuat!")
                        else:
                            st.error("❌ Gagal membuat file ZIP")
                
                # Individual file downloads
                st.write("**Atau download file individual:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not rules.empty:
                        # Convert rules to CSV
                        rules_csv = rules.copy()
                        rules_csv['antecedents'] = rules_csv['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_csv['consequents'] = rules_csv['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        csv = rules_csv.to_csv(index=False)
                        st.download_button(
                            label="📄 Download Aturan Asosiasi (CSV)",
                            data=csv,
                            file_name="association_rules.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if not frequent_itemsets.empty:
                        # Convert itemsets to CSV
                        itemsets_csv = frequent_itemsets.copy()
                        itemsets_csv['itemsets'] = itemsets_csv['itemsets'].apply(lambda x: ', '.join(list(x)))
                        
                        csv = itemsets_csv.to_csv(index=False)
                        st.download_button(
                            label="📄 Download Itemset Sering (CSV)",
                            data=csv,
                            file_name="frequent_itemsets.csv",
                            mime="text/csv"
                        )
        else:
            st.warning("Tidak ada aturan yang tersedia untuk visualisasi. Coba sesuaikan parameter.")
    
    # Kesimpulan dan Rekomendasi Penempatan Barang
    if not rules.empty:
        st.header("🎯 Kesimpulan dan Rekomendasi Penempatan Barang")
        
        # Filter rules dengan lift > 1 (asosiasi positif) dan confidence yang tinggi
        strong_rules = rules[
            (rules['lift'] > 1.0) & 
            (rules['confidence'] >= min_confidence)
        ].sort_values(['lift', 'confidence'], ascending=[False, False])
        
        if not strong_rules.empty:
            st.success(f"✅ Ditemukan {len(strong_rules)} aturan asosiasi kuat untuk rekomendasi penempatan barang")
            
            # Buat tabel rekomendasi
            st.subheader("📋 Tabel Rekomendasi Penempatan Barang")
            
            # Format data untuk tabel
            recommendations = []
            for idx, rule in strong_rules.iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                
                # Tentukan prioritas berdasarkan lift dan confidence
                if rule['lift'] >= 3.0 and rule['confidence'] >= 0.5:
                    priority = "🔴 Tinggi"
                    strategy = "Letakkan sangat berdekatan"
                elif rule['lift'] >= 2.0 and rule['confidence'] >= 0.3:
                    priority = "🟡 Sedang"
                    strategy = "Letakkan berdekatan"
                else:
                    priority = "🟢 Rendah"
                    strategy = "Letakkan dalam area yang sama"
                
                recommendations.append({
                    'Barang A (Trigger)': antecedents,
                    'Barang B (Target)': consequents,
                    'Lift': f"{rule['lift']:.2f}",
                    'Confidence': f"{rule['confidence']*100:.1f}%" if display_format == "Persentase (%)" else f"{rule['confidence']:.3f}",
                    'Support': f"{rule['support']*100:.2f}%" if display_format == "Persentase (%)" else f"{rule['support']:.4f}",
                    'Prioritas': priority,
                    'Strategi Penempatan': strategy
                })
            
            # Tampilkan tabel
            recommendations_df = pd.DataFrame(recommendations)
            st.dataframe(recommendations_df, use_container_width=True)
            
            # Tambahkan penjelasan
            st.markdown("""
            ### 📊 Penjelasan Tabel:
            
            **Prioritas:**
            - 🔴 **Tinggi**: Asosiasi sangat kuat, sangat direkomendasikan untuk diletakkan berdekatan
            - 🟡 **Sedang**: Asosiasi cukup kuat, direkomendasikan untuk diletakkan berdekatan
            - 🟢 **Rendah**: Asosiasi lemah, cukup diletakkan dalam area yang sama
            
            **Metrik:**
            - **Lift**: Semakin tinggi nilai lift (>1), semakin kuat asosiasi antar barang
            - **Confidence**: Persentase kemungkinan barang B dibeli jika barang A sudah dibeli
            - **Support**: Persentase transaksi yang mengandung kombinasi barang tersebut
            """)
            
            # Tambahkan insights tambahan
            st.subheader("💡 Insights Tambahan")
            
            # Analisis barang yang paling sering menjadi trigger
            trigger_items = []
            for rule in strong_rules.iterrows():
                antecedents = list(rule[1]['antecedents'])
                for item in antecedents:
                    trigger_items.append(item)
            
            if trigger_items:
                trigger_counts = Counter(trigger_items)
                top_triggers = trigger_counts.most_common(5)
                
                st.write("**Barang yang paling sering menjadi 'trigger' pembelian:**")
                for item, count in top_triggers:
                    st.write(f"• {item}: {count} kali")
            
            # Analisis barang yang paling sering menjadi target
            target_items = []
            for rule in strong_rules.iterrows():
                consequents = list(rule[1]['consequents'])
                for item in consequents:
                    target_items.append(item)
            
            if target_items:
                target_counts = Counter(target_items)
                top_targets = target_counts.most_common(5)
                
                st.write("**Barang yang paling sering menjadi 'target' pembelian:**")
                for item, count in top_targets:
                    st.write(f"• {item}: {count} kali")
            
            # Rekomendasi strategi umum
            st.subheader("🎯 Strategi Penempatan Umum")
            st.markdown("""
            1. **Cross-Merchandising**: Letakkan barang dengan prioritas tinggi berdekatan
            2. **End-Cap Displays**: Gunakan ujung rak untuk kombinasi barang dengan lift tinggi
            3. **Bundling**: Pertimbangkan untuk membuat paket khusus untuk kombinasi yang sangat kuat
            4. **Promotional Adjacency**: Letakkan barang dengan asosiasi kuat di area promosi
            5. **Seasonal Placement**: Sesuaikan penempatan dengan musim atau event tertentu
            """)
            
        else:
            st.warning("⚠️ Tidak ditemukan aturan asosiasi yang cukup kuat untuk rekomendasi penempatan barang.")
            st.info("💡 Tips: Coba turunkan nilai minimum confidence atau lift untuk mendapatkan lebih banyak rekomendasi.")
    
    else:
        st.warning("⚠️ Tidak ada aturan asosiasi yang tersedia untuk membuat kesimpulan.")
        st.info("💡 Tips: Coba sesuaikan parameter support, confidence, dan lift untuk mendapatkan aturan asosiasi.")



except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.info("Silakan periksa format data Anda dan coba lagi.")
