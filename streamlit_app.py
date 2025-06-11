# enhanced_streamlit_app.py - Complete plant disease detection app

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="🌱 Plant Disease Doctor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disease information database
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "common_name": "Pepper Bacterial Spot",
        "severity_factors": ["leaf_coverage", "fruit_damage", "plant_age"],
        "treatments": [
            "Apply copper-based bactericide spray",
            "Remove infected plant parts immediately",
            "Improve air circulation around plants",
            "Avoid overhead watering",
            "Use drip irrigation instead"
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Rotate crops annually",
            "Maintain proper plant spacing",
            "Disinfect tools between plants"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Pepper__bell___healthy": {
        "common_name": "Healthy Pepper Plant",
        "severity_factors": [],
        "treatments": [
            "Continue current care routine",
            "Monitor for early disease signs",
            "Maintain proper watering schedule"
        ],
        "prevention": [
            "Regular inspection for pests",
            "Balanced fertilization",
            "Proper pruning techniques"
        ],
        "severity_ranges": {"healthy": 1.0}
    },
    "Potato___Early_blight": {
        "common_name": "Potato Early Blight",
        "severity_factors": ["lesion_size", "leaf_yellowing", "stem_damage"],
        "treatments": [
            "Apply fungicide (chlorothalonil or mancozeb)",
            "Remove affected lower leaves",
            "Ensure adequate potassium levels",
            "Improve soil drainage",
            "Space plants for better air circulation"
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Crop rotation with non-solanaceous plants",
            "Avoid overhead irrigation",
            "Hill potatoes properly"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Potato___Late_blight": {
        "common_name": "Potato Late Blight",
        "severity_factors": ["water_soaked_lesions", "white_growth", "tuber_rot"],
        "treatments": [
            "Apply systemic fungicide immediately",
            "Remove all infected plant material",
            "Harvest unaffected tubers quickly",
            "Destroy infected plants (don't compost)",
            "Monitor weather conditions closely"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Avoid overhead watering",
            "Ensure good drainage",
            "Monitor weather forecasts for blight conditions"
        ],
        "severity_ranges": {"mild": 0.5, "moderate": 0.7, "severe": 1.0}
    },
    "Potato___healthy": {
        "common_name": "Healthy Potato Plant",
        "severity_factors": [],
        "treatments": [
            "Maintain current care practices",
            "Regular soil moisture monitoring",
            "Continue balanced fertilization"
        ],
        "prevention": [
            "Weekly disease monitoring",
            "Proper hilling techniques",
            "Adequate spacing between plants"
        ],
        "severity_ranges": {"healthy": 1.0}
    },
    "Tomato_Bacterial_spot": {
        "common_name": "Tomato Bacterial Spot",
        "severity_factors": ["leaf_spots", "fruit_lesions", "defoliation"],
        "treatments": [
            "Apply copper-based spray at first sign",
            "Remove infected leaves and fruit",
            "Increase plant spacing",
            "Use drip irrigation only",
            "Apply mulch to prevent soil splash"
        ],
        "prevention": [
            "Use resistant varieties",
            "Start with certified disease-free seeds",
            "Avoid working with wet plants",
            "Sanitize tools regularly"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Tomato_Early_blight": {
        "common_name": "Tomato Early Blight",
        "severity_factors": ["target_spots", "lower_leaf_damage", "stem_lesions"],
        "treatments": [
            "Apply fungicide spray (chlorothalonil)",
            "Remove lower infected leaves",
            "Improve air circulation",
            "Stake plants properly",
            "Avoid overhead watering"
        ],
        "prevention": [
            "Mulch around plants",
            "Rotate crops for 3-4 years",
            "Provide adequate nutrition",
            "Water at soil level"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Tomato_Late_blight": {
        "common_name": "Tomato Late Blight",
        "severity_factors": ["water_soaked_spots", "white_fungal_growth", "rapid_spread"],
        "treatments": [
            "Apply systemic fungicide immediately",
            "Remove all infected plant parts",
            "Improve ventilation drastically",
            "Harvest green tomatoes if severe",
            "Consider removing entire plant"
        ],
        "prevention": [
            "Choose resistant varieties",
            "Avoid overhead irrigation",
            "Monitor humidity levels",
            "Space plants widely"
        ],
        "severity_ranges": {"mild": 0.5, "moderate": 0.7, "severe": 1.0}
    },
    "Tomato_Leaf_Mold": {
        "common_name": "Tomato Leaf Mold",
        "severity_factors": ["yellow_spots", "fuzzy_growth", "humidity_level"],
        "treatments": [
            "Reduce humidity immediately",
            "Increase ventilation",
            "Apply fungicide if severe",
            "Remove affected leaves",
            "Water only at soil level"
        ],
        "prevention": [
            "Ensure good air circulation",
            "Avoid overhead watering",
            "Use resistant varieties",
            "Monitor greenhouse humidity"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Tomato_Septoria_leaf_spot": {
        "common_name": "Tomato Septoria Leaf Spot",
        "severity_factors": ["circular_spots", "black_centers", "defoliation_rate"],
        "treatments": [
            "Apply fungicide with mancozeb",
            "Remove infected lower leaves",
            "Mulch heavily around plants",
            "Water at ground level only",
            "Improve plant spacing"
        ],
        "prevention": [
            "Crop rotation for 3+ years",
            "Use drip irrigation",
            "Stake plants for air flow",
            "Clean garden debris"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "common_name": "Two-Spotted Spider Mites",
        "severity_factors": ["stippling_damage", "webbing_presence", "leaf_yellowing"],
        "treatments": [
            "Spray with miticide or insecticidal soap",
            "Increase humidity around plants",
            "Use predatory mites if available",
            "Remove heavily infested leaves",
            "Spray undersides of leaves thoroughly"
        ],
        "prevention": [
            "Regular leaf inspection",
            "Maintain adequate humidity",
            "Avoid over-fertilizing with nitrogen",
            "Encourage beneficial insects"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Tomato__Target_Spot": {
        "common_name": "Tomato Target Spot",
        "severity_factors": ["concentric_rings", "leaf_yellowing", "fruit_lesions"],
        "treatments": [
            "Apply fungicide with chlorothalonil",
            "Remove infected plant debris",
            "Improve air circulation",
            "Avoid overhead irrigation",
            "Stake plants properly"
        ],
        "prevention": [
            "Crop rotation schedule",
            "Clean cultivation practices",
            "Resistant variety selection",
            "Proper plant spacing"
        ],
        "severity_ranges": {"mild": 0.6, "moderate": 0.8, "severe": 1.0}
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "common_name": "Tomato Yellow Leaf Curl Virus",
        "severity_factors": ["leaf_curling", "yellowing_pattern", "stunting"],
        "treatments": [
            "Remove infected plants immediately",
            "Control whitefly vectors with insecticides",
            "Use reflective mulches",
            "Install fine mesh screens",
            "No chemical cure available"
        ],
        "prevention": [
            "Use virus-resistant varieties",
            "Control whitefly populations",
            "Remove weeds that harbor virus",
            "Use yellow sticky traps"
        ],
        "severity_ranges": {"mild": 0.4, "moderate": 0.7, "severe": 1.0}
    },
    "Tomato__Tomato_mosaic_virus": {
        "common_name": "Tomato Mosaic Virus",
        "severity_factors": ["mosaic_pattern", "leaf_distortion", "fruit_quality"],
        "treatments": [
            "Remove infected plants immediately",
            "Sanitize tools with bleach solution",
            "Control aphid vectors",
            "No chemical treatment available",
            "Prevent spread to healthy plants"
        ],
        "prevention": [
            "Use certified virus-free seeds",
            "Control aphid populations",
            "Avoid tobacco use near plants",
            "Practice good sanitation"
        ],
        "severity_ranges": {"mild": 0.4, "moderate": 0.7, "severe": 1.0}
    },
    "Tomato_healthy": {
        "common_name": "Healthy Tomato Plant",
        "severity_factors": [],
        "treatments": [
            "Continue current care routine",
            "Maintain consistent watering",
            "Regular pruning of suckers"
        ],
        "prevention": [
            "Weekly plant inspection",
            "Balanced fertilization schedule",
            "Proper staking and support"
        ],
        "severity_ranges": {"healthy": 1.0}
    }
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try to load the model
        if os.path.exists('plant_disease_model.h5'):
            model = tf.keras.models.load_model('plant_disease_model.h5')
            st.success("✅ AI model loaded successfully!")
            return model
        elif os.path.exists('plant_disease_model.keras'):
            model = tf.keras.models.load_model('plant_disease_model.keras')
            st.success("✅ AI model loaded successfully!")
            return model
        else:
            st.warning("⚠️ No trained model found. Using demo mode.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load class names"""
    try:
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except:
        # Default class names if file doesn't exist
        return list(DISEASE_INFO.keys())

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to model input size
    image = image.resize((224, 224))
    # Convert to array
    image_array = np.array(image)
    # Normalize
    image_array = image_array.astype('float32') / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def assess_severity(disease_class, confidence):
    """Assess disease severity based on confidence and disease type"""
    if disease_class not in DISEASE_INFO:
        return "Unknown", "gray"
    
    disease_data = DISEASE_INFO[disease_class]
    
    # Healthy plants
    if "healthy" in disease_class.lower():
        return "Healthy", "green"
    
    # Disease severity assessment
    severity_ranges = disease_data["severity_ranges"]
    
    if confidence < list(severity_ranges.values())[0]:
        return "Mild", "yellow"
    elif confidence < list(severity_ranges.values())[1] if len(severity_ranges) > 1 else 0.8:
        return "Moderate", "orange"
    else:
        return "Severe", "red"

def create_confidence_chart(predictions, class_names):
    """Create confidence visualization"""
    df = pd.DataFrame({
        'Disease': [DISEASE_INFO.get(class_names[i], {}).get('common_name', class_names[i]) for i in range(len(predictions))],
        'Confidence': predictions * 100
    })
    df = df.sort_values('Confidence', ascending=True).tail(5)
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Disease', 
        orientation='h',
        title="Top 5 Predictions",
        color='Confidence',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=300)
    return fig

def create_severity_gauge(severity_text, confidence):
    """Create severity gauge"""
    color_map = {"Healthy": "green", "Mild": "yellow", "Moderate": "orange", "Severe": "red"}
    color = color_map.get(severity_text, "gray")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Level<br><span style='font-size:0.8em;color:{color}'>{severity_text}</span>"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.title("🌱 Plant Disease Doctor")
    st.markdown("### AI-Powered Plant Disease Detection & Treatment Recommendations")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Upload section
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the plant leaf or affected area"
        )
        
        # Settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Minimum confidence level for diagnosis"
        )
        
        show_top_n = st.selectbox(
            "Show Top Predictions",
            options=[3, 5, 7],
            index=1
        )
        
        # Model status
        st.subheader("🤖 Model Status")
        model = load_model()
        class_names = load_class_names()
        
        if model:
            st.success(f"Model loaded: {len(class_names)} diseases")
        else:
            st.warning("Demo mode active")
    
    # Main content
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Plant Image", use_column_width=True)
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 AI Analysis")
            
            if model is not None:
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Make prediction
                with st.spinner("🧠 AI is analyzing your plant..."):
                    predictions = model.predict(processed_image, verbose=0)[0]
                
                # Get top prediction
                top_prediction_idx = np.argmax(predictions)
                top_confidence = predictions[top_prediction_idx]
                predicted_class = class_names[top_prediction_idx]
                
                if top_confidence > confidence_threshold:
                    # Disease information
                    disease_info = DISEASE_INFO.get(predicted_class, {})
                    common_name = disease_info.get('common_name', predicted_class)
                    
                    # Assess severity
                    severity, severity_color = assess_severity(predicted_class, top_confidence)
                    
                    # Display main result
                    st.success(f"**Detected:** {common_name}")
                    st.metric(
                        label="Confidence Level",
                        value=f"{top_confidence:.1%}",
                        delta=f"Severity: {severity}"
                    )
                    
                    # Severity gauge
                    gauge_fig = create_severity_gauge(severity, top_confidence)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                else:
                    st.warning(f"⚠️ **Unknown detected** - Confidence too low ({top_confidence:.1%})")
                    st.info("Try uploading a clearer image or adjust the confidence threshold")
            
            else:
                # Demo mode
                st.warning("🎭 **Demo Mode** - Upload an image to test the interface")
                predicted_class = "Tomato_Early_blight"  # Demo prediction
                top_confidence = 0.85
                severity = "Moderate"
        
        # Detailed Analysis Section
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        # Create tabs for different information
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Diagnosis", "💊 Treatment", "🛡️ Prevention", "📈 Confidence"])
        
        with tab1:
            if model and top_confidence > confidence_threshold:
                disease_info = DISEASE_INFO.get(predicted_class, {})
                common_name = disease_info.get('common_name', predicted_class)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {common_name}")
                    
                    if "healthy" in predicted_class.lower():
                        st.success("🌿 **Great news!** Your plant appears healthy.")
                        st.markdown("Continue your current care routine and monitor regularly.")
                    else:
                        severity, color = assess_severity(predicted_class, top_confidence)
                        st.markdown(f"**Severity Level:** <span style='color:{color}; font-weight:bold'>{severity}</span>", unsafe_allow_html=True)
                        
                        if severity == "Severe":
                            st.error("🚨 **Immediate action required!** This condition can spread rapidly.")
                        elif severity == "Moderate":
                            st.warning("⚠️ **Treatment recommended** to prevent spread.")
                        else:
                            st.info("ℹ️ **Early stage detected** - Good time for intervention.")
                
                with col2:
                    # Plant type icon
                    if "tomato" in predicted_class.lower():
                        st.markdown("🍅 **Tomato Plant**")
                    elif "potato" in predicted_class.lower():
                        st.markdown("🥔 **Potato Plant**")
                    elif "pepper" in predicted_class.lower():
                        st.markdown("🌶️ **Pepper Plant**")
                    
                    st.metric("Confidence", f"{top_confidence:.1%}")
            else:
                st.info("Upload a plant image for detailed diagnosis")
        
        with tab2:
            if model and top_confidence > confidence_threshold:
                disease_info = DISEASE_INFO.get(predicted_class, {})
                treatments = disease_info.get('treatments', [])
                
                if treatments:
                    st.markdown("### 💊 Recommended Treatments")
                    
                    for i, treatment in enumerate(treatments, 1):
                        if i == 1:
                            st.markdown(f"**🥇 Priority:** {treatment}")
                        else:
                            st.markdown(f"**{i}.** {treatment}")
                    
                    # Urgency indicator
                    severity, _ = assess_severity(predicted_class, top_confidence)
                    if severity == "Severe":
                        st.error("⏰ **Act within 24-48 hours** to prevent further damage")
                    elif severity == "Moderate":
                        st.warning("⏰ **Treat within 1 week** for best results")
                    else:
                        st.info("⏰ **Monitor and treat when convenient**")
                else:
                    st.success("🌿 No treatment needed - plant is healthy!")
            else:
                st.info("Diagnosis needed for treatment recommendations")
        
        with tab3:
            if model and top_confidence > confidence_threshold:
                disease_info = DISEASE_INFO.get(predicted_class, {})
                prevention = disease_info.get('prevention', [])
                
                if prevention:
                    st.markdown("### 🛡️ Prevention Strategies")
                    
                    for tip in prevention:
                        st.markdown(f"• {tip}")
                    
                    st.markdown("### 🌱 General Plant Health Tips")
                    st.markdown("""
                    - **Regular Inspection:** Check plants weekly for early signs
                    - **Proper Spacing:** Ensure good air circulation
                    - **Clean Tools:** Sanitize between plants
                    - **Crop Rotation:** Change plant families annually
                    - **Soil Health:** Maintain proper drainage and nutrients
                    """)
            else:
                st.info("Diagnosis needed for prevention recommendations")
        
        with tab4:
            if model:
                # Confidence visualization
                confidence_fig = create_confidence_chart(predictions, class_names)
                st.plotly_chart(confidence_fig, use_container_width=True)
                
                # Top predictions table
                st.markdown("### 📋 All Predictions")
                
                # Create detailed predictions table
                pred_data = []
                for i, conf in enumerate(predictions):
                    disease_name = DISEASE_INFO.get(class_names[i], {}).get('common_name', class_names[i])
                    pred_data.append({
                        'Disease': disease_name,
                        'Confidence': f"{conf:.1%}",
                        'Status': '✅ Detected' if conf > confidence_threshold else '❌ Below threshold'
                    })
                
                pred_df = pd.DataFrame(pred_data)
                pred_df = pred_df.sort_values('Confidence', ascending=False)
                st.dataframe(pred_df, use_container_width=True)
            else:
                st.info("Model needed for confidence analysis")
    
    else:
        # Welcome screen
        st.markdown("""
        ### 🌟 Welcome to Plant Disease Doctor!
        
        **How it works:**
        1. 📱 Upload a clear photo of your plant
        2. 🤖 AI analyzes the image for diseases
        3. 📋 Get diagnosis, treatment, and prevention tips
        
        **Supported Plants:**
        - 🍅 Tomatoes (8 diseases + healthy)
        - 🥔 Potatoes (3 diseases + healthy) 
        - 🌶️ Peppers (2 diseases + healthy)
        
        **Tips for best results:**
        - Use good lighting
        - Focus on affected areas
        - Include leaves and symptoms clearly
        - Avoid blurry or distant shots
        """)
        
        # Sample images
        st.markdown("### 📸 Example Images")
        sample_cols = st.columns(3)
        
        with sample_cols[0]:
            st.markdown("**✅ Good Image**")
            st.markdown("Clear, focused, good lighting")
        
        with sample_cols[1]:
            st.markdown("**⚠️ Acceptable Image**")
            st.markdown("Visible symptoms, adequate quality")
        
        with sample_cols[2]:
            st.markdown("**❌ Poor Image**")
            st.markdown("Blurry, dark, or too distant")

if __name__ == "__main__":
    main()