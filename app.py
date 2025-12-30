import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import json
import requests
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'ollama_available' not in st.session_state:
    st.session_state.ollama_available = False

# ============= UTILITY FUNCTIONS =============

def check_ollama_connection():
    """Check if Ollama is running locally"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def detect_column_types(df):
    """Detect column types: numeric, categorical, datetime, text"""
    column_types = {}
    
    for col in df.columns:
        non_null = df[col].dropna()
        
        if len(non_null) == 0:
            column_types[col] = 'empty'
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'datetime'
        else:
            unique_ratio = len(non_null.unique()) / len(non_null)
            if unique_ratio < 0.5 or len(non_null.unique()) < 20:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
    
    return column_types

# ============= BUILT-IN ANALYSIS FUNCTIONS =============

def analyze_outliers(df, column_types):
    """Detect outliers using Z-score method"""
    numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
    outliers = []
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) > 0:
            z_scores = np.abs(stats.zscore(values))
            outlier_indices = np.where(z_scores > 3)[0]
            
            for idx in outlier_indices:
                outliers.append({
                    'Column': col,
                    'Row Index': values.index[idx],
                    'Value': values.iloc[idx],
                    'Z-Score': z_scores[idx]
                })
    
    return pd.DataFrame(outliers) if outliers else None

def analyze_statistics(df, column_types):
    """Calculate comprehensive statistics for numeric columns"""
    numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
    
    stats_data = []
    for col in numeric_cols:
        values = df[col].dropna()
        stats_data.append({
            'Column': col,
            'Count': len(values),
            'Mean': values.mean(),
            'Median': values.median(),
            'Std Dev': values.std(),
            'Min': values.min(),
            'Max': values.max(),
            'Q1': values.quantile(0.25),
            'Q3': values.quantile(0.75),
            'IQR': values.quantile(0.75) - values.quantile(0.25)
        })
    
    return pd.DataFrame(stats_data)

def analyze_missing_values(df):
    """Analyze missing values in dataset"""
    missing_data = []
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_data.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing %': (missing_count / len(df)) * 100
            })
    
    return pd.DataFrame(missing_data) if missing_data else None

def analyze_categorical(df, column_types):
    """Analyze categorical columns"""
    categorical_cols = [col for col, typ in column_types.items() if typ == 'categorical']
    
    results = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        results[col] = {
            'Unique Values': len(value_counts),
            'Most Common': value_counts.index[0] if len(value_counts) > 0 else None,
            'Most Common Count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'Distribution': value_counts.head(10).to_dict()
        }
    
    return results

def analyze_correlation(df, column_types):
    """Calculate correlation matrix for numeric columns"""
    numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
    
    if len(numeric_cols) < 2:
        return None
    
    return df[numeric_cols].corr()

def analyze_duplicates(df):
    """Find duplicate rows"""
    duplicates = df[df.duplicated(keep=False)]
    return duplicates if len(duplicates) > 0 else None

def query_ollama(prompt, df_info):
    """Query Ollama for custom analysis"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": f"""You are a data analyst. Based on this dataset information:
                
Columns: {df_info['columns']}
Column Types: {df_info['types']}
Shape: {df_info['shape']}
Sample Data:
{df_info['sample']}

User Query: {prompt}

Provide a clear analysis or recommendation. Be specific and actionable.""",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get('response', 'No response generated')
        else:
            return "Error: Could not get response from Ollama"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# ============= MAIN APP =============

# Header
st.markdown('<p class="main-header">📊 AI Data Analyst Pro</p>', unsafe_allow_html=True)
st.markdown("**Upload data, run built-in analyses, or use custom queries with Ollama**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported formats: CSV, Excel, JSON"
    )
    
    if uploaded_file:
        try:
            # Load data based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                st.session_state.data = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                st.session_state.data = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                st.session_state.data = pd.read_json(uploaded_file)
            
            st.success(f"✅ Loaded {len(st.session_state.data)} rows")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    st.divider()
    
    # Ollama status
    st.subheader("🤖 Ollama Status")
    if st.button("Check Ollama Connection", use_container_width=True):
        st.session_state.ollama_available = check_ollama_connection()
    
    if st.session_state.ollama_available:
        st.success("✅ Ollama is running")
    else:
        st.warning("⚠️ Ollama not detected")
        with st.expander("How to install Ollama"):
            st.markdown("""
            **For Custom Queries, install Ollama:**
            
            1. Visit: https://ollama.ai
            2. Download for your OS
            3. Install and run: `ollama run llama2`
            4. Refresh this page
            
            This enables AI-powered custom queries!
            """)
    
    st.divider()
    
    # Download options
    if st.session_state.data is not None:
        st.subheader("💾 Export Options")
        
        # CSV download
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            "📄 Download as CSV",
            csv,
            "data_export.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Excel download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state.data.to_excel(writer, index=False)
        
        st.download_button(
            "📊 Download as Excel",
            buffer.getvalue(),
            "data_export.xlsx",
            "application/vnd.ms-excel",
            use_container_width=True
        )

# Main content
if st.session_state.data is None:
    st.info("👆 Upload a data file from the sidebar to get started")
    
    # Sample data option
    st.subheader("🎯 Or try with sample data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Sales Sample", use_container_width=True):
            st.session_state.data = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=100),
                'Product': np.random.choice(['Widget', 'Gadget', 'Doohickey'], 100),
                'Sales': np.random.randint(100, 1000, 100),
                'Profit': np.random.randint(20, 200, 100),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
            })
            st.rerun()
    
    with col2:
        if st.button("Load Customer Sample", use_container_width=True):
            st.session_state.data = pd.DataFrame({
                'Customer_ID': range(1, 101),
                'Age': np.random.randint(18, 80, 100),
                'Income': np.random.randint(30000, 150000, 100),
                'Purchases': np.random.randint(1, 50, 100),
                'Category': np.random.choice(['Bronze', 'Silver', 'Gold'], 100)
            })
            st.rerun()

else:
    df = st.session_state.data
    column_types = detect_column_types(df)
    
    # Dataset Overview
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:1rem;">Total Rows</h3>
            <h2 style="margin:0.5rem 0 0 0;">{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:1rem;">Total Columns</h3>
            <h2 style="margin:0.5rem 0 0 0;">{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_count = sum(1 for t in column_types.values() if t == 'numeric')
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:1rem;">Numeric Columns</h3>
            <h2 style="margin:0.5rem 0 0 0;">{numeric_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        categorical_count = sum(1 for t in column_types.values() if t == 'categorical')
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:1rem;">Categorical Columns</h3>
            <h2 style="margin:0.5rem 0 0 0;">{categorical_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Column Types
    st.subheader("🔍 Detected Column Types")
    col_type_df = pd.DataFrame([
        {'Column': col, 'Type': typ} 
        for col, typ in column_types.items()
    ])
    st.dataframe(col_type_df, use_container_width=True, hide_index=True)
    
    # Data Preview
    with st.expander("👁️ Preview Data", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)
    
    st.divider()
    
    # Analysis Tabs
    st.header("🔬 Analysis Tools")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Built-in Analyses", 
        "🤖 Custom Query (Ollama)", 
        "📈 Visualizations",
        "📜 History"
    ])
    
    # TAB 1: Built-in Analyses
    with tab1:
        st.subheader("Select Analysis Type")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            if st.button("🎯 Find Outliers", use_container_width=True):
                with st.spinner("Analyzing outliers..."):
                    result = analyze_outliers(df, column_types)
                    if result is not None and not result.empty:
                        st.success(f"Found {len(result)} outliers")
                        st.dataframe(result, use_container_width=True)
                        
                        # Visualization
                        if len(result) > 0:
                            fig = px.scatter(
                                result, 
                                x='Column', 
                                y='Z-Score',
                                color='Column',
                                title='Outliers by Column',
                                hover_data=['Value']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No outliers detected (Z-score > 3)")
            
            if st.button("📊 Statistical Summary", use_container_width=True):
                with st.spinner("Calculating statistics..."):
                    result = analyze_statistics(df, column_types)
                    if not result.empty:
                        st.success("Statistics calculated")
                        st.dataframe(result, use_container_width=True)
                    else:
                        st.info("No numeric columns found")
            
            if st.button("❓ Missing Values", use_container_width=True):
                with st.spinner("Analyzing missing values..."):
                    result = analyze_missing_values(df)
                    if result is not None:
                        st.warning(f"Found missing values in {len(result)} columns")
                        st.dataframe(result, use_container_width=True)
                        
                        # Visualization
                        fig = px.bar(
                            result, 
                            x='Column', 
                            y='Missing %',
                            title='Missing Values by Column (%)',
                            color='Missing %',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ No missing values detected!")
        
        with analysis_col2:
            if st.button("🏷️ Categorical Analysis", use_container_width=True):
                with st.spinner("Analyzing categories..."):
                    result = analyze_categorical(df, column_types)
                    if result:
                        st.success(f"Analyzed {len(result)} categorical columns")
                        for col, stats in result.items():
                            with st.expander(f"📊 {col}"):
                                st.write(f"**Unique Values:** {stats['Unique Values']}")
                                st.write(f"**Most Common:** {stats['Most Common']} ({stats['Most Common Count']} times)")
                                st.write("**Top 10 Distribution:**")
                                dist_df = pd.DataFrame(
                                    list(stats['Distribution'].items()),
                                    columns=['Value', 'Count']
                                )
                                st.dataframe(dist_df, use_container_width=True)
                    else:
                        st.info("No categorical columns found")
            
            if st.button("🔗 Correlation Matrix", use_container_width=True):
                with st.spinner("Calculating correlations..."):
                    result = analyze_correlation(df, column_types)
                    if result is not None:
                        st.success("Correlation matrix calculated")
                        fig = px.imshow(
                            result,
                            title='Correlation Heatmap',
                            color_continuous_scale='RdBu',
                            zmin=-1,
                            zmax=1
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 2 numeric columns")
            
            if st.button("🔄 Find Duplicates", use_container_width=True):
                with st.spinner("Searching for duplicates..."):
                    result = analyze_duplicates(df)
                    if result is not None:
                        st.warning(f"Found {len(result)} duplicate rows")
                        st.dataframe(result.head(50), use_container_width=True)
                    else:
                        st.success("✅ No duplicate rows found!")
    
    # TAB 2: Custom Query with Ollama
    with tab2:
        if not st.session_state.ollama_available:
            st.warning("⚠️ Ollama is not running. Install it to enable custom queries.")
            st.markdown("""
            ### 🚀 Get Started with Ollama
            
            1. **Download**: Visit https://ollama.ai
            2. **Install**: Follow instructions for your OS
            3. **Run**: Execute `ollama run llama2` in terminal
            4. **Return**: Come back and click "Check Ollama Connection"
            
            Ollama enables AI-powered custom analysis of your data!
            """)
        else:
            st.success("✅ Ollama is connected and ready!")
            
            custom_query = st.text_area(
                "Ask anything about your data:",
                placeholder="e.g., What insights can you provide about sales trends?\nWhich columns should I focus on for prediction?\nAre there any unusual patterns?",
                height=100
            )
            
            if st.button("🚀 Run Custom Query", use_container_width=True, type="primary"):
                if custom_query.strip():
                    with st.spinner("Querying Ollama..."):
                        df_info = {
                            'columns': list(df.columns),
                            'types': column_types,
                            'shape': df.shape,
                            'sample': df.head(3).to_string()
                        }
                        
                        response = query_ollama(custom_query, df_info)
                        
                        st.markdown("### 🤖 AI Response")
                        st.markdown(f"""
                        <div class="analysis-card">
                        {response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'type': 'Custom Query',
                            'query': custom_query,
                            'result': response
                        })
                else:
                    st.error("Please enter a query")
    
    # TAB 3: Visualizations
    with tab3:
        st.subheader("Create Custom Visualizations")
        
        viz_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"]
        )
        
        if viz_type == "Histogram":
            numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                bins = st.slider("Number of Bins", 10, 100, 30)
                
                fig = px.histogram(df, x=selected_col, nbins=bins, title=f'Distribution of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                fig = px.box(df, y=selected_col, title=f'Box Plot of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot":
            numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
            if len(numeric_cols) >= 2:
                col1_sel = st.selectbox("X-axis", numeric_cols)
                col2_sel = st.selectbox("Y-axis", numeric_cols, index=1)
                
                categorical_cols = [col for col, typ in column_types.items() if typ == 'categorical']
                color_col = st.selectbox("Color by (optional)", ['None'] + categorical_cols)
                
                fig = px.scatter(
                    df, 
                    x=col1_sel, 
                    y=col2_sel,
                    color=None if color_col == 'None' else color_col,
                    title=f'{col1_sel} vs {col2_sel}'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            categorical_cols = [col for col, typ in column_types.items() if typ == 'categorical']
            if categorical_cols:
                selected_col = st.selectbox("Select Column", categorical_cols)
                
                value_counts = df[selected_col].value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f'Top Values in {selected_col}',
                    labels={'x': selected_col, 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                fig = px.line(df.reset_index(), x='index', y=selected_col, title=f'{selected_col} Over Rows')
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: History
    with tab4:
        st.subheader("📜 Analysis History")
        
        if st.session_state.analysis_history:
            for idx, item in enumerate(reversed(st.session_state.analysis_history)):
                with st.expander(f"{item['timestamp']} - {item['type']}"):
                    st.write(f"**Query:** {item['query']}")
                    st.write(f"**Result:** {item['result']}")
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.analysis_history = []
                st.rerun()
        else:
            st.info("No analysis history yet. Run some analyses to see them here!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>AI Data Analyst Pro</strong> | Built with Streamlit & Ollama</p>
    <p>Built-in analyses are free • Custom queries require Ollama</p>
</div>
""", unsafe_allow_html=True)