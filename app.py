import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import os
from openai import OpenAI
from langdetect import detect
from deep_translator import GoogleTranslator
import datetime
import math
import html

st.set_page_config(
    page_title="Material Price Prediction and ChatBot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# lock sidebar expanded
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

from dotenv import load_dotenv
load_dotenv()

from blob_reader import get_latest_csv_df

from utils import load_css, load_js, inject_script
# setting page layout

print("Page Config Set")
print(f"{st.session_state.get('initial_sidebar_state')}")

# Load external CSS and JavaScript files
load_css('app.css')
load_js('app.js')

# Initialize mobile detection
inject_script('mobile_detection')

col1, col2 = st.columns([2.8, 1.2]) 


connection_string = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")
folder_path = os.getenv("AZURE_FOLDER_PATH")

# Call the function
df = get_latest_csv_df(connection_string,container_name,folder_path)

print(df.head())

df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors="coerce")


# Rename columns
df.rename(
    columns={
        "share_price": "TASI",
        "FEDFUND_rate": "Interest_rate"
    },
    inplace=True
)
print(df.head())

# ------ Filter the Data
df = df.sort_values(by="Date") #sorting so that plot is consitent
filtered_df=df
# Drop rows where year is invalid
df = df.dropna(subset=["year"])

# Get max year
max_year = int(df["year"].max())

cutoff_date = pd.to_datetime(f"{max_year}-01-01")
# Filter using datetime
other_factors_df = df[df['Date'] >= cutoff_date]
other_factors_df = other_factors_df[other_factors_df['Type'] == 'Actual_price']
# If you want final display as yyyy-MM-dd (string)
other_factors_df['Date'] = other_factors_df['Date'].dt.strftime("%Y-%m-%d")
# OR keep the last occurrence
other_factors_df = other_factors_df.drop_duplicates(subset=["Date"], keep="last")
# ------ End of the Data Filter


# ----- Charts in 2/3rd part from left --------- #
with col1:
    with st.sidebar:
        st.sidebar.header("Filters")
        print(f"initial_sidebar_state: {st.session_state.get('initial_sidebar_state')}")
        # 1. Category Filter
        Category_options = filtered_df["Category"].unique().tolist()
        selected_category = st.selectbox("Select category:", Category_options)
        # 2. Material Filter (dependent on Category)
        material_options = filtered_df[filtered_df["Category"] == selected_category]["Material"].unique().tolist()
        selected_material = st.selectbox("Select Material:", material_options)
        # 3. Year Filter
        min_year = filtered_df["Date"].dt.year.min()
        max_year = filtered_df["Date"].dt.year.max()
        
        current_year = datetime.datetime.now().year
        Year_options = sorted(filtered_df["Date"].dt.year.unique().tolist())

        # If current year is not in your dataset (future year), default to last available
        default_index = Year_options.index(current_year) if current_year in Year_options else len(Year_options)-1

        selected_year = st.sidebar.selectbox(
            "Select year:", 
            Year_options,
            index=default_index
        )

        # Y-axis range multiplier for the main chart (constant value, no UI)
        y_range_multiplier = 5.0

    # Filter Data Based on User Selections
    filtered_df = filtered_df[(filtered_df["Material"] == selected_material) & (filtered_df["Date"].dt.year == selected_year)]

    # Split actual & predicted
    actual = filtered_df[filtered_df["Type"]=="Actual_price"]
    pred = filtered_df[filtered_df["Type"]=="Predicted_price"]

#------------------## Chart 1: Line Chart with Border------------------------------------#

    # Build figure
    fig1 = go.Figure()

    # Plot the "Present" (actual) data
    if not actual.empty:
        fig1.add_trace(go.Scatter(
            x=actual["Date"],
            y=actual["Material_price"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="green", width=3)
        ))

    # Plot the "Predicted" data ONLY if it exists in the filtered DataFrame
    if not pred.empty:
        # To ensure a continuous line, add the last 'actual' point to the 'predicted' data

        cutoff_date = actual["Date"].max()
        pred_only = pred[pred["Date"] > cutoff_date]
        # Bridge point (last actual) + predictions

        # Create bridge for continuity (last actual + predictions)
        bridge_x = pd.concat([actual.iloc[[-1]]["Date"], pred_only["Date"]])
        bridge_y = pd.concat([actual.iloc[[-1]]["Material_price"], pred_only["Material_price"]])

        # Add prediction line (no marker for bridge point)
        fig1.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="Prediction",
            line=dict(color="orange", width=3, dash="dot"),
            marker=dict(color="orange", size=6),
            # Hide marker for the first (bridge) point
            marker_symbol="circle",
            marker_size=[0] + [6] * (len(bridge_y)-1)  
        ))

        # Buffer (±10%) only for future predictions
        upper = pred_only["Material_price"] * 1.05
        lower = pred_only["Material_price"] * 0.95

        fig1.add_trace(go.Scatter(
            x=pred_only["Date"],
            y=upper,
            line=dict(width=0),
            marker=dict(color="red", size=6),
            showlegend=False
        ))

        fig1.add_trace(go.Scatter(
            x=pred_only["Date"],
            y=lower,
            line=dict(width=0),
            marker=dict(color="red", size=6),
            fill="tonexty",
            fillcolor="rgba(255,165,0,0.2)",
            name="±5% Buffer"
        ))

    # Add the vertical divider line ONLY if actual data exists
    if selected_year == max_year and not pred.empty:
        cutoff_date = actual["Date"].max()
        fig1.add_vline(
            x=cutoff_date,
            line_width=2,
            line_dash="dash",
            line_color="gray",
        )

    # Streamlit App
    st.title(f"Price Index Prediction for {selected_material}")

    # Calculate dynamic y-axis range by expanding the exact data range using sidebar multiplier
    try:
        combined_prices = pd.concat([
            actual["Material_price"] if not actual.empty else pd.Series(dtype=float),
            pred["Material_price"] if not pred.empty else pd.Series(dtype=float)
        ])
        if not combined_prices.empty:
            min_val = float(combined_prices.min())
            max_val = float(combined_prices.max())
            # center-based expansion: keep center and multiply half-range
            center = (min_val + max_val) / 2.0
            half_range = (max_val - min_val) / 2.0
            if half_range == 0:
                # fallback when all values equal: expand by a small absolute amount
                half_range = max(1.0, abs(center) * 0.05)
            expanded_half = half_range * float(y_range_multiplier)
            y_min = center - expanded_half
            y_max = center + expanded_half
            # Compute a "nice" tick step so ticks are at round values and not too dense
            try:
                desired_ticks = 6;
                raw_step = (y_max - y_min) / float(desired_ticks);
                if raw_step <= 0:
                    step = max(1.0, abs(center) * 0.05);
                else:
                    # magnitude and nice multiples (1,2,5,10)
                    mag = 10 ** math.floor(math.log10(raw_step));
                    for m in (1, 2, 5, 10):
                        step = m * mag;
                        if step >= raw_step:
                            break;

                # compute nice axis bounds aligned to step
                nice_min = math.floor(y_min / step) * step
                nice_max = math.ceil(y_max / step) * step

                # determine tick label format (integer when step >= 1)
                if step >= 1:
                    tickformat = ".0f"
                else:
                    decimals = max(0, -int(math.floor(math.log10(step))))
                    tickformat = f".{decimals}f"

                fig1.update_yaxes(range=[nice_min, nice_max], dtick=step, tickformat=tickformat)
            except Exception:
                fig1.update_yaxes(range=[y_min, y_max])
    except Exception:
        # if anything fails, skip setting custom range
        pass

    # Layout
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Material Price",
        template="plotly_white",  # keep base style
        plot_bgcolor="white",     # white chart background
        paper_bgcolor="white",    # white paper background
        # Manually create the legend entries
        showlegend=True,
        xaxis=dict(
        tickformat="%b",
        dtick = "M1"
        ), # check this x axis date format
        legend=dict(
            traceorder="normal",
            itemsizing="constant"
        )
    )

    st.plotly_chart(fig1, use_container_width=True)

    import plotly.graph_objects as go
    import streamlit as st

    st.header("External Factors")

    factors_to_plot = ["oil_price", "gold_price", "TASI", "Interest_rate"]
    colors = {
        "oil_price": "#FF6600",   # Orange
        "gold_price": "#FFD700",  # Gold
        "TASI": "#1E90FF",        # Blue
        "Interest_rate": "#2E8B57" # Green
    }

    # Loop through factors in chunks of 2
    for i in range(0, len(factors_to_plot), 2):
        cols = st.columns(2, gap="large")
        for j, factor in enumerate(factors_to_plot[i:i+2]):
            with cols[j]:
                fig = go.Figure()
                if not other_factors_df.empty:
                    fig.add_trace(go.Scatter(
                        x=other_factors_df["Date"],
                        y=other_factors_df[factor],
                        mode="lines+markers",
                        line=dict(color=colors[factor], width=3, shape="spline"),
                        marker=dict(size=7, color=colors[factor], line=dict(width=1, color="black")),
                        hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>"
                    ))

                # Create proper title with TASI capitalized
                if factor == "TASI":
                    chart_title = "TASI"
                else:
                    chart_title = factor.replace('_', ' ').title()
                
                fig.update_layout(
                    title=chart_title,
                    xaxis_title="",
                    yaxis_title="",
                    template="plotly_white",
                    showlegend=False,
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=20),
                    plot_bgcolor="white",     # white chart background
                    paper_bgcolor="white",    # white paper background
                    font=dict(size=12, family="Arial"),
                    xaxis=dict(tickformat="%b"),
                )

                fig.update_xaxes(showgrid=False, tickangle=45)
                fig.update_yaxes(showgrid=True, zeroline=False)

                st.plotly_chart(fig, use_container_width=True)

### ---------------------- ChatBot Starts Here --------------------------- ###

# Step 2: Set the environment variables for Azure AI Search
# These variables configure the search service and index for retrieving documents

AZURE_AI_SEARCH_SERVICE_NAME = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME", "")
AZURE_AI_SEARCH_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "")
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

# Validate required variables
missing_vars = []
if not AZURE_AI_SEARCH_SERVICE_NAME:
    missing_vars.append("AZURE_AI_SEARCH_SERVICE_NAME")
if not AZURE_AI_SEARCH_INDEX_NAME:
    missing_vars.append("AZURE_AI_SEARCH_INDEX_NAME")
if not AZURE_AI_SEARCH_API_KEY:
    missing_vars.append("AZURE_AI_SEARCH_API_KEY")
if not AZURE_OPENAI_API_KEY:
    missing_vars.append("AZURE_OPENAI_API_KEY")
if not AZURE_OPENAI_ENDPOINT:
    missing_vars.append("AZURE_OPENAI_ENDPOINT")

if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

#----------------langchain imports-------------------------------#

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever

# ---------------- Page Setup ---------------- #
# Note: Page config already set at the top of the file

#------------------------ Step 1: Initialize the AzureAI Search Retriever -------------------------------------------#

retriever = AzureAISearchRetriever(
    api_key=AZURE_AI_SEARCH_API_KEY,
    service_name=AZURE_AI_SEARCH_SERVICE_NAME,
    index_name=AZURE_AI_SEARCH_INDEX_NAME,
    content_key="content",
    top_k=3
)

#----------------------------------- Step 2: Define the prompt template for the language model-----------------------------#
# This sets up how the context and question will be formatted for the model

prompt = ChatPromptTemplate.from_template(
    """
You are an assistant that answers material price questions using the dataset only (2013-2025, actual & predicted values).

Rules:
- Always answer in English.
- Use only the provided context. Do not invent data.
- For greetings like hi/Hi/hello/hey, always respond consistently.
- Always include other relevant factors (gold, oil, shares) if available.
- If a price for a month is missing, estimate it from nearby months and explain clearly that it is an estimate.
- If no data is found at all, reply: "Sorry, I dont have information about that."

Context:
{context}

Question:
{question}

Answer:
"""
)

#-------------------Step 3: Initialize the Azure Chat OpenAI model------------------------------#
# This sets up the model to be used for generating responses

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    model="gpt-4.1",
    api_version="2024-12-01-preview",
    temperature=0.7,
    max_tokens=250
)

#----------------- Step 4: Create a processing chain-------------------------------------#
# This chain will process the retrieved context and the user question
chain = (
    {"context": retriever , "question": RunnablePassthrough()}  # Set context using the retriever and format it
    | prompt                                                               # Pass the formatted context and question to the prompt
    | llm                                                                  # Generate a response using the language model
    | StrOutputParser()                                                   # Parse the output to a string format
)
def send_message(user_input):
    if user_input.strip():
        user_lang = detect(user_input)
        query_text = GoogleTranslator(source="ar", target="en").translate(user_input) if user_lang == "ar" else user_input

        # ---- Replace this with your LLM call ----
        response = chain.invoke(query_text)

        if user_lang == "ar":
            response = GoogleTranslator(source="en", target="ar").translate(response)

        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] =[]

if "temp_input" not in st.session_state:
    st.session_state["temp_input"] = " "

with col2:

    st.markdown("""
    <div id= "fixed-chatbot-inner" style="display: flex; position: fixed; flex-direction: column; height: 100vh; border: 1px solid #ddd; border-radius: 12px; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background-color: #ffffff;">
    
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 12px; margin-bottom: 10px; '>
        <h1 style='text-align: left; font-size: clamp(18px, 4vw, 30px); margin: 0; word-wrap: break-word; color: #262730;'>Construction Price Index AI Consultant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Render chat messages only if there is chat history; otherwise hide the message area
    if st.session_state.get("chat_history") and len(st.session_state.chat_history) > 0:
        # Make the chat area flex:1 and add bottom padding so messages don't hide behind the input
        chat_html = """
            <div style="display:flex; justify-content:center; width:100%;">
                <div class='chat-area' id='chat-box'
                    style='flex:1; overflow-y:auto; padding-bottom:300px;
                            height: calc(100vh - 200px); max-height:72vh; width: 500px;
                            max-width:650px;'>
            """
        for chat in st.session_state.chat_history:
            safe_text = html.escape(chat["content"]).replace("\n", "<br>")
            if chat["role"] == "user":
                chat_html += f"<div class='chat-message user'><div class='chat-bubble'>{safe_text}</div></div>"
            else:
                chat_html += f"<div class='chat-message assistant'><div class='chat-bubble'>{safe_text}</div></div>"

        chat_html += """
            </div>
        </div>
        """
        st.markdown(chat_html, unsafe_allow_html=True)

    else:
        # empty placeholder area so layout stays consistent when no messages
        st.markdown("<div class='chat-area' style='flex:1; min-height:200px; padding-bottom:140px; width : 500px;'></div>", unsafe_allow_html=True)
    
    # Use a form with columns for horizontal layout
    with st.form(key="chat_form", clear_on_submit=True):
        
        col_input, col_button = st.columns([5, 1])
        
        with col_input:
            # Dynamic placeholder based on chat history
            if st.session_state.get("chat_history") and len(st.session_state.chat_history) > 0:
                placeholder_text = "Ask about material prices..."
            else:
                placeholder_text = "Start a Conversation with AI Consultant..."

            user_input = st.text_area(
                "Message",
                value="",
                placeholder=placeholder_text,
                key="user_input_form",
                label_visibility="collapsed",
                height=55
            )
        with col_button:
            submitted = st.form_submit_button("→", use_container_width=True)
        if submitted and user_input.strip():
            send_message(user_input.strip())
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get("chat_history") and len(st.session_state.chat_history) > 0:
        inject_script('auto_scroll')

    # Initialize textarea auto-resize
    inject_script('textarea_resize')