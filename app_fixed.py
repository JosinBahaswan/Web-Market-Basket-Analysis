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

# Set page configuration
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# Title and description
st.title("Market Basket Analysis - Apriori Algorithm")
st.markdown("""
This application performs market basket analysis using the Apriori algorithm to discover 
associations between products frequently purchased together.
""")

# File uploader with default to Data_Transaksi.csv
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

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
        st.write(f"Columns found: {df.columns.tolist()}")
        st.write(f"Number of rows: {len(df)}")
        
        # Show a sample of the data
        st.write("Sample of the loaded data:")
        st.write(df.head(3))
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to prepare transactions
@st.cache_data
def prepare_transactions(df):
    # Convert the Products column to a list of lists
    if 'Products' in df.columns:
        # Show raw product data for debugging
        st.write("Raw product data (first 3 rows):")
        for i in range(min(3, len(df))):
            st.write(f"Row {i+1}: {df['Products'].iloc[i]}")
        
        # Check if the data is already in the correct format
        sample = df['Products'].iloc[0] if not df.empty else ""
        
        if isinstance(sample, str):
            # Split by comma and strip whitespace
            transactions = []
            for products in df['Products']:
                if isinstance(products, str):
                    # Split by comma and strip whitespace
                    items = [item.strip() for item in products.split(',')]
                    transactions.append(items)
                else:
                    transactions.append([])
        else:
            st.error("Products column does not contain string data")
            return []
        
        # Debug info
        st.write(f"Number of transactions: {len(transactions)}")
        st.write("Sample of processed transactions:")
        st.write(transactions[:3])
        
        return transactions
    else:
        st.error(f"'Products' column not found. Available columns: {df.columns.tolist()}")
        return []

# Function to create one-hot encoded DataFrame
@st.cache_data
def create_one_hot_df(transactions):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Debug info
    st.write(f"One-hot encoded data shape: {df_encoded.shape}")
    st.write(f"Number of unique products: {len(df_encoded.columns)}")
    
    return df_encoded

# Function to apply Apriori algorithm
@st.cache_data
def apply_apriori(df_encoded, min_support):
    try:
        # Ensure min_support is not too high
        if min_support > 0.5:
            st.warning("Support threshold is very high. Lowering to 0.01.")
            min_support = 0.01
        
        # Check if df_encoded is valid
        if df_encoded.empty:
            st.error("Encoded dataframe is empty. Cannot apply Apriori algorithm.")
            return pd.DataFrame(columns=['support', 'itemsets'])
        
        # Display sample of encoded data
        st.write("Sample of encoded data (first 3 rows, first 5 columns):")
        sample_cols = min(5, len(df_encoded.columns))
        st.write(df_encoded.iloc[:3, :sample_cols])
        
        # Apply apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # Debug info
        if len(frequent_itemsets) == 0:
            st.warning(f"No frequent itemsets found with support threshold {min_support}. Try lowering the support value.")
        else:
            st.success(f"Found {len(frequent_itemsets)} frequent itemsets.")
            
        return frequent_itemsets
    except Exception as e:
        st.error(f"Error in Apriori algorithm: {e}")
        return pd.DataFrame(columns=['support', 'itemsets'])

# Function to generate association rules
@st.cache_data
def generate_rules(frequent_itemsets, min_threshold):
    try:
        if len(frequent_itemsets) == 0:
            st.warning("No frequent itemsets available to generate rules.")
            return pd.DataFrame()
        
        # Check if the frequent_itemsets DataFrame has the required format
        if 'support' not in frequent_itemsets.columns or 'itemsets' not in frequent_itemsets.columns:
            st.error("Frequent itemsets dataframe does not have the required columns (support, itemsets).")
            return pd.DataFrame()
        
        # Display sample of frequent itemsets
        st.write("Sample of frequent itemsets:")
        st.write(frequent_itemsets.head(3))
        
        # Generate rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
        
        # Debug info
        if len(rules) == 0:
            st.warning(f"No rules found with confidence threshold {min_threshold}. Try lowering the confidence value.")
        else:
            st.success(f"Generated {len(rules)} association rules.")
            
        return rules
    except Exception as e:
        st.error(f"Error generating association rules: {str(e)}")
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
        title='Top 15 Most Frequent Items',
        color='Frequency',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    return item_df

# Function to visualize association rules network
def plot_association_network(rules, min_lift=1, max_rules=20):
    if len(rules) == 0:
        st.warning("No rules to visualize. Try lowering the minimum confidence or lift.")
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
        st.warning("No rules to visualize. Try lowering the minimum confidence or lift.")
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
        title=f'Top {max_rules} Association Rules by {metric.capitalize()}',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Function to create scatter plot of rules
def plot_rules_scatter(rules):
    if len(rules) == 0:
        st.warning("No rules to visualize. Try lowering the minimum confidence or lift.")
        return
    
    # Create scatter plot
    fig = px.scatter(
        rules, x='support', y='confidence', 
        size='lift', color='lift',
        hover_name=[f"{', '.join(list(x))} → {', '.join(list(y))}" for x, y in zip(rules['antecedents'], rules['consequents'])],
        log_x=True, size_max=20,
        color_continuous_scale='Viridis',
        title='Association Rules - Support vs Confidence'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main application logic
try:
    # Load data
    if uploaded_file is not None:
        df = load_data(uploaded_file=uploaded_file)
    else:
        df = load_data(file_path='Data_Transaksi.csv')
    
    if df is None or df.empty:
        st.error("Failed to load data or data is empty.")
        st.stop()
    
    # Display raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)
    
    # Prepare transactions
    transactions = prepare_transactions(df)
    
    if not transactions:
        st.error("No transactions to analyze.")
        st.stop()
    
    # Display transaction sample
    with st.expander("View Sample Transactions"):
        st.write(transactions[:5])
    
    # Sidebar for parameter display format
    st.sidebar.header("Parameter Display Format")
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
    with st.spinner("Applying Apriori algorithm..."):
        frequent_itemsets = apply_apriori(df_encoded, min_support)
    
    # Display frequent itemsets
    st.header("Frequent Itemsets")
    st.write(f"Number of frequent itemsets: {len(frequent_itemsets)}")
    
    with st.expander("View Frequent Itemsets"):
        if not frequent_itemsets.empty:
            # Format support as percentage if needed
            if display_format == "Persentase (%)":
                frequent_itemsets_display = frequent_itemsets.copy()
                frequent_itemsets_display['support'] = frequent_itemsets_display['support'] * 100
                frequent_itemsets_display['itemsets'] = frequent_itemsets_display['itemsets'].apply(lambda x: ', '.join(list(x)))
                frequent_itemsets_display.rename(columns={'support': 'support (%)'}, inplace=True)
                st.dataframe(frequent_itemsets_display)
            else:
                st.dataframe(frequent_itemsets)
        else:
            st.warning("No frequent itemsets found. Try lowering the support threshold.")
    
    # Generate association rules
    with st.spinner("Generating association rules..."):
        if not frequent_itemsets.empty:
            rules = generate_rules(frequent_itemsets, min_confidence)
            
            # Filter by lift
            if not rules.empty:
                rules = rules[rules['lift'] >= min_lift]
        else:
            rules = pd.DataFrame()
    
    # Display association rules
    st.header("Association Rules")
    st.write(f"Number of rules: {len(rules)}")
    
    with st.expander("View Association Rules"):
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
            st.warning("No association rules found. Try adjusting the parameters.")
    
    # Visualizations
    if not frequent_itemsets.empty:
        st.header("Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Frequent Items", "Association Network", "Rules Heatmap", "Rules Scatter Plot"])
        
        with tab1:
            st.subheader("Frequent Items")
            item_df = plot_frequent_items(transactions)
        
        if not rules.empty:
            with tab2:
                st.subheader("Association Network")
                plot_association_network(rules, min_lift=min_lift)
            
            with tab3:
                st.subheader("Rules Heatmap")
                plot_rules_heatmap(rules)
            
            with tab4:
                st.subheader("Rules Scatter Plot")
                plot_rules_scatter(rules)
        else:
            st.warning("No rules available for visualization. Try adjusting the parameters.")
    else:
        st.warning("No frequent itemsets available for visualization. Try lowering the support threshold.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please check your data format and try again.")
