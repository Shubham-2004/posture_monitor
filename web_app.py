import streamlit as st
import numpy as np
import time
import os
from dotenv import load_dotenv
import re
import math

# More robust cloud detection
is_streamlit_cloud = (
    os.getenv("STREAMLIT_CLOUD", "false").lower() == "true" or
    "STREAMLIT_SHARING_PORT" in os.environ or
    "STREAMLIT_SERVER_PORT" in os.environ or
    "STREAMLIT_RUN_ON_SAVE" in os.environ
)

# Try to conditionally import OpenCV
try:
    import cv2
    import mediapipe as mp
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    
# Set cloud mode based on environment and CV availability
CLOUD_MODE = is_streamlit_cloud or not CV_AVAILABLE

if CLOUD_MODE:
    st.warning("Running in cloud mode with limited functionality. For full experience, run locally.")

# Import remaining modules
from agno.tools.email import EmailTools
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    st.error("GROQ API key not found. Please add it to your environment variables or .env file.")

# Setup agent for news search
try:
    search_agent = Agent(
        model=Groq(id="qwen-qwq-32b"),
        tools=[GoogleSearchTools],
        description="Web search agent",
        markdown=True,
        show_tool_calls=True,
        debug_mode=True,
    )
except Exception as e:
    st.error(f"Failed to initialize search agent: {str(e)}")
    search_agent = None

# Helper to format news results - show only health-related links
def format_news(news_raw):
    # Extract only the URLs from markdown-style links
    items = re.findall(r'- \[(.*?)\]\((.*?)\)', news_raw)
    if not items:
        return ''  # show nothing if no links

    health_keywords = ['health', 'posture', 'ergonomic', 'spine', 'back', 'neck', 'wellness', 'pain']

    return ''.join(
        f"<a href='{url}' target='_blank'>{title}</a><br><small>{url}</small><br><br>"
        for title, url in items
        if any(
            keyword in title.lower() or keyword in url.lower()
            for keyword in health_keywords
        )
    )

# Format posture benefits for UI display
def format_posture_benefits():
    benefits = [
        "Reduces back and neck pain by properly distributing weight",
        "Improves breathing and circulation by creating more space for lungs",
        "Decreases abnormal wearing of joint surfaces that could result in arthritis",
        "Reduces the stress on ligaments holding the joints of the spine together",
        "Prevents the spine from becoming fixed in abnormal positions"
    ]
    formatted = (
        "<div style='background-color: #1E1E1E; color: #E0E0E0; padding: 15px; border-radius: 10px; margin: 15px 0;'>"
        + "<h3 style='color: #4CAF50; margin-bottom: 10px;'>Benefits of Good Posture:</h3><ul style='margin-left: 20px;'>"
    ) 
    for benefit in benefits:
        formatted += f"<li style='margin-bottom: 5px;'>{benefit}</li>"

    formatted += "</ul></div>"
    return formatted

def findDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def findAngle(x1, y1, x2, y2):
    if y1 == 0:
        return 0
    try:
        theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
        return int(180/math.pi * theta)
    except:
        return 0

# Create an interactive demo for cloud mode
def create_interactive_demo():
    # Create two demo images for good and bad posture
    good_posture = np.zeros((480, 640, 3), dtype=np.uint8)
    bad_posture = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Common drawing parameters
    color = (200, 200, 200)
    thickness = 2
    
    # Draw head (same position for both)
    center = (320, 100)
    radius = 30
    for i in range(radius - thickness, radius + thickness):
        for angle in range(0, 360):
            x = int(center[0] + i * np.cos(np.radians(angle)))
            y = int(center[1] + i * np.sin(np.radians(angle)))
            if 0 <= x < good_posture.shape[1] and 0 <= y < good_posture.shape[0]:
                good_posture[y, x] = color
                bad_posture[y, x] = color
    
    # Draw body - straight for good posture, bent for bad posture
    # Good posture body
    pt1_good = (320, 130)
    pt2_good = (320, 300)
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt1_good[0] + t * (pt2_good[0] - pt1_good[0]) + i)
            y = int(pt1_good[1] + t * (pt2_good[1] - pt1_good[1]))
            if 0 <= x < good_posture.shape[1] and 0 <= y < good_posture.shape[0]:
                good_posture[y, x] = color
    
    # Bad posture body (leaning forward)
    pt1_bad = (320, 130)
    pt2_bad = (340, 300)  # leaning forward
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt1_bad[0] + t * (pt2_bad[0] - pt1_bad[0]) + i)
            y = int(pt1_bad[1] + t * (pt2_bad[1] - pt1_bad[1]))
            if 0 <= x < bad_posture.shape[1] and 0 <= y < bad_posture.shape[0]:
                bad_posture[y, x] = color
    
    # Draw arms
    # Good posture arms
    pt1 = (250, 180)
    pt2 = (390, 180)
    pt3 = (320, 180)
    # Left arm
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt3[0] + t * (pt1[0] - pt3[0]))
            y = int(pt3[1] + t * (pt1[1] - pt3[1]) + i)
            if 0 <= x < good_posture.shape[1] and 0 <= y < good_posture.shape[0]:
                good_posture[y, x] = color
                bad_posture[y, x] = color
    # Right arm
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt3[0] + t * (pt2[0] - pt3[0]))
            y = int(pt3[1] + t * (pt2[1] - pt3[1]) + i)
            if 0 <= x < good_posture.shape[1] and 0 <= y < good_posture.shape[0]:
                good_posture[y, x] = color
                bad_posture[y, x] = color
    
    # Draw legs
    # Good posture legs
    pt1 = (270, 400)
    pt2 = (370, 400)
    pt3_good = (320, 300)
    # Left leg
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt3_good[0] + t * (pt1[0] - pt3_good[0]) + i)
            y = int(pt3_good[1] + t * (pt1[1] - pt3_good[1]))
            if 0 <= x < good_posture.shape[1] and 0 <= y < good_posture.shape[0]:
                good_posture[y, x] = color
    # Right leg
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt3_good[0] + t * (pt2[0] - pt3_good[0]) + i)
            y = int(pt3_good[1] + t * (pt2[1] - pt3_good[1]))
            if 0 <= x < good_posture.shape[1] and 0 <= y < good_posture.shape[0]:
                good_posture[y, x] = color
    
    # Bad posture legs
    pt3_bad = (340, 300)  # leaning forward
    # Left leg
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt3_bad[0] + t * (pt1[0] - pt3_bad[0]) + i)
            y = int(pt3_bad[1] + t * (pt1[1] - pt3_bad[1]))
            if 0 <= x < bad_posture.shape[1] and 0 <= y < bad_posture.shape[0]:
                bad_posture[y, x] = color
    # Right leg
    for i in range(-thickness, thickness):
        for t in np.linspace(0, 1, 100):
            x = int(pt3_bad[0] + t * (pt2[0] - pt3_bad[0]) + i)
            y = int(pt3_bad[1] + t * (pt2[1] - pt3_bad[1]))
            if 0 <= x < bad_posture.shape[1] and 0 <= y < bad_posture.shape[0]:
                bad_posture[y, x] = color
    
    # Add angle labels to good posture
    # Draw neck angle line for good posture (green)
    good_posture[100:180, 320:322] = (0, 255, 0)  # vertical line
    # Draw torso angle line for good posture (green)
    good_posture[180:300, 320:322] = (0, 255, 0)  # vertical line
    
    # Add angle labels to bad posture
    # Draw neck angle line for bad posture (red)
    for i in range(100, 180):
        x = int(320 + (i-100) * 0.3)
        if 0 <= x < bad_posture.shape[1] and 0 <= i < bad_posture.shape[0]:
            bad_posture[i, x:x+2] = (0, 0, 255)
    
    # Draw torso angle line for bad posture (red)
    for i in range(180, 300):
        x = int(320 + (i-180) * 0.1)
        if 0 <= x < bad_posture.shape[1] and 0 <= i < bad_posture.shape[0]:
            bad_posture[i, x:x+2] = (0, 0, 255)
    
    # Add text labels
    good_text = "GOOD POSTURE"
    bad_text = "BAD POSTURE"
    
    # Add text to both images
    if CV_AVAILABLE:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(good_posture, good_text, (250, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(bad_posture, bad_text, (260, 30), font, 0.7, (0, 0, 255), 2)
        
        # Add angle text
        cv2.putText(good_posture, "Neck angle: 10°", (10, 30), font, 0.6, (0, 255, 0), 2)
        cv2.putText(good_posture, "Torso angle: 5°", (10, 60), font, 0.6, (0, 255, 0), 2)
        
        cv2.putText(bad_posture, "Neck angle: 35°", (10, 30), font, 0.6, (0, 0, 255), 2)
        cv2.putText(bad_posture, "Torso angle: 15°", (10, 60), font, 0.6, (0, 0, 255), 2)
    
    return good_posture, bad_posture

def main():
    st.set_page_config(page_title="AI Posture Monitor", layout="wide")
    
    # Two columns for layout - main content and sidebar
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🧑‍💻 AI Posture Monitor")
        st.write("Monitor your posture in real-time and receive alerts when needed")
        
        # User email input with validation
        user_email = st.text_input("Enter your email to receive posture alerts:", "")
        if user_email and not re.match(r"[^@]+@[^@]+\.[^@]+", user_email):
            st.error("Please enter a valid email address")
            user_email = ""
            
        # Posture threshold sliders for customization
        st.subheader("Posture Settings")
        cols = st.columns(3)
        with cols[0]:
            neck_threshold = st.slider("Neck Angle Threshold", 15, 40, 25)
        with cols[1]:
            torso_threshold = st.slider("Torso Angle Threshold", 5, 20, 10)
        with cols[2]:
            alert_time = st.slider("Alert After (seconds)", 5, 30, 15)
            
        start_btn = st.button('Start Monitoring', type="primary")
        
        # Camera feed placeholder
        FRAME_WINDOW = st.image([])
        status_text = st.empty()
        alert_text = st.empty()
        
        # Stats display
        stats_cols = st.columns(3)
        good_time_display = stats_cols[0].empty()
        bad_time_display = stats_cols[1].empty()
        posture_quality = stats_cols[2].empty()
        
        # Placeholder for posture benefits that appear after bad posture
        posture_benefits_container = st.empty()

    # Sidebar content
    st.sidebar.title("Health Resources")
    
    # Sidebar for news
    st.sidebar.header("Latest Health News")
    with st.sidebar:
        if search_agent:
            with st.spinner("Fetching health news..."):
                try:
                    news_response = search_agent.run("Find the latest news about posture health, ergonomics, and back pain prevention")
                    news_raw = news_response.text if hasattr(news_response, 'text') else str(news_response)
                    news_content = format_news(news_raw)
                    if news_content:
                        st.markdown(news_content, unsafe_allow_html=True)
                    else:
                        st.info("No relevant health news found. Please try again later.")
                except Exception as e:
                    st.error(f"Unable to fetch news: {str(e)}")
                    st.info("Try refreshing the page or check your API key.")
        else:
            st.info("News search is currently unavailable.")

    # Add a sidebar section for quick posture tips
    st.sidebar.header("Quick Posture Tips")
    st.sidebar.info("\n".join([
        "- Keep your back straight and shoulders relaxed.",
        "- Adjust your chair height so feet are flat on floor.",
        "- Position your screen at eye level.",
        "- Take regular breaks (5 min every hour).",
        "- Keep elbows close to body when typing.",
        "- Avoid slouching or leaning forward."
    ]))
    
    # Add posture exercise section
    st.sidebar.header("5-Min Desk Exercises")
    st.sidebar.success("\n".join([
        "1. **Neck Rolls**: Slowly roll your neck in a circle (5× each direction)",
        "2. **Shoulder Rolls**: Roll shoulders backwards (10×)",
        "3. **Chest Stretch**: Clasp hands behind back, squeeze shoulder blades (15s)",
        "4. **Wrist Stretch**: Extend arm, gently pull fingers back (10s each hand)",
        "5. **Seated Spinal Twist**: Twist torso to look behind you (10s each side)"
    ]))

    if not user_email or not start_btn:
        with col1:
            st.info("Please enter your email and click 'Start Monitoring' to begin.")
        st.stop()

    # Email agent setup
    try:
        sender_email = "shubhambera2004@gmail.com"
        sender_name = "Posture Monitor"
        sender_passkey = "trvk ibsq yexz njab"  # Store securely in .env in production
        email_agent = Agent(
            model=Groq(id="qwen-qwq-32b"),
            tools=[
                EmailTools(
                    receiver_email=user_email,
                    sender_email=sender_email,
                    sender_name=sender_name,
                    sender_passkey=sender_passkey,
                )
            ]
        )
    except Exception as e:
        st.error(f"Failed to initialize email agent: {str(e)}")
        email_agent = None

    # Check if we're in cloud mode - if so, run interactive demo mode
    if CLOUD_MODE:
        st.info("🌟 **Interactive Demo Mode** 🌟 - For full camera functionality, run this app locally.")
        
        # Create demo images
        good_posture, bad_posture = create_interactive_demo()
        
        # Create placeholders for interactive demo
        demo_image = st.empty()
        demo_status = st.empty()
        
        # Interactive demo settings
        posture_type = st.radio("Select posture to visualize:", ["Good Posture", "Bad Posture", "Toggle (Demo)"])
        
        if posture_type == "Good Posture":
            demo_image.image(good_posture)
            demo_status.success("👍 Good Posture: Spine aligned, neck straight")
            
            # Show stats for good posture
            good_time_display.metric("Good Posture Time", "120.0s")
            bad_time_display.metric("Bad Posture Time", "0.0s")
            posture_quality.metric("Posture Quality", "100.0%")
            
        elif posture_type == "Bad Posture":
            demo_image.image(bad_posture)
            demo_status.error("👎 Bad Posture: Forward head, hunched shoulders")
            
            # Show stats for bad posture
            good_time_display.metric("Good Posture Time", "0.0s")
            bad_time_display.metric("Bad Posture Time", "25.5s")
            posture_quality.metric("Posture Quality", "0.0%")
            
            # Show posture benefits
            posture_benefits_container.markdown(
                format_posture_benefits(),
                unsafe_allow_html=True
            )
            
        else:  # Toggle mode
            # Create a loop that simulates posture detection
            placeholder = st.empty()
            placeholder.info("Starting demo in 3 seconds...")
            time.sleep(1)
            placeholder.info("Starting demo in 2 seconds...")
            time.sleep(1)
            placeholder.info("Starting demo in 1 second...")
            time.sleep(1)
            placeholder.empty()
            
            # Initialize counters
            cycle_count = 0
            good_time = 0
            bad_time = 0
            total_time = 0
            current_state = "good"  # Start with good posture
            
            # Run for 6 cycles (about 1 minute)
            while cycle_count < 6:
                # Show good posture for 5 seconds
                demo_image.image(good_posture)
                demo_status.success("👍 Good Posture: Spine aligned, neck straight")
                posture_benefits_container.empty()  # Clear benefits display
                
                # Update stats
                for i in range(5):
                    good_time += 1
                    total_time += 1
                    quality = (good_time / total_time) * 100 if total_time > 0 else 100
                    
                    good_time_display.metric("Good Posture Time", f"{good_time}.0s")
                    bad_time_display.metric("Bad Posture Time", f"{bad_time}.0s")
                    posture_quality.metric("Posture Quality", f"{round(quality, 1)}%")
                    
                    # Short sleep
                    time.sleep(1)
                
                # Show bad posture for 5 seconds
                demo_image.image(bad_posture)
                demo_status.error("👎 Bad Posture: Forward head, hunched shoulders")
                
                # Update stats
                for i in range(5):
                    bad_time += 1
                    total_time += 1
                    quality = (good_time / total_time) * 100 if total_time > 0 else 0
                    
                    good_time_display.metric("Good Posture Time", f"{good_time}.0s")
                    bad_time_display.metric("Bad Posture Time", f"{bad_time}.0s")
                    posture_quality.metric("Posture Quality", f"{round(quality, 1)}%")
                    
                    # Show benefits after 3 seconds of bad posture
                    if i == 2:
                        posture_benefits_container.markdown(
                            format_posture_benefits(),
                            unsafe_allow_html=True
                        )
                    
                    # Show alert after 4 seconds
                    if i == 4 and email_agent:
                        alert_text.warning(f"⚠️ Email alert would be sent to {user_email} for bad posture!")
                    
                    # Short sleep
                    time.sleep(1)
                
                cycle_count += 1
                
            # After demo loop completes
            st.success("Demo completed! In the full version, continuous monitoring would occur.")
            
        st.write("""
        ### How this works in the full version:
        1. The app uses your webcam to track key points on your body
        2. It calculates neck and torso angles in real-time
        3. If angles exceed thresholds for too long, you get an email alert
        4. You can customize sensitivity and timing of alerts
        
        Run this app locally on your computer for the full experience with camera access!
        """)
        
        st.stop()
        
    # Local mode with camera access continues below...
    
    # The rest of your code remains the same for the local camera version
    # [rest of your existing code for camera mode]
    
    # Try to access the camera with better error handling
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera. The app will run in demo mode with a static image instead.")
            # Use a dummy image for demonstration
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, "Camera not available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            FRAME_WINDOW.image(image)
            
            # Show posture benefits in demo mode
            posture_benefits_container.markdown(
                format_posture_benefits(),
                unsafe_allow_html=True
            )
            
            st.info("In a real environment, the app would track your posture and provide real-time feedback.")
            st.stop()
    except Exception as e:
        st.error(f"Error accessing camera: {str(e)}")
        st.info("The app will run in demo mode with a static image instead.")
        # Use a dummy image for demonstration
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(image, "Camera not available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        FRAME_WINDOW.image(image)
        
        # Show posture benefits in demo mode
        posture_benefits_container.markdown(
            format_posture_benefits(),
            unsafe_allow_html=True
        )
        
        st.info("In a real environment, the app would track your posture and provide real-time feedback.")
        st.stop()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (127, 255, 0)
    red = (50, 50, 255)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    white = (255, 255, 255)
    good_frames = 0
    bad_frames = 0
    bad_posture_alert_sent = False
    benefits_displayed = False
    fps = 30
    
    # Store posture history for trend analysis
    posture_history = []
    
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            status_text.warning("Webcam not found or disconnected.")
            break
            
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(img_rgb)
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        
        if lm is None:
            FRAME_WINDOW.image(image, channels="BGR")
            status_text.info("No person detected. Please position yourself in front of the camera.")
            continue
            
        # Draw skeleton connections for better visualization
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image, 
            lm, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
            
        # Use the same keypoints as your reference
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        
        # Calculate posture metrics
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        
        # Posture logic with custom thresholds
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Shoulders aligned', (w - 280, 30), font, 0.6, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Shoulders not aligned', (w - 280, 30), font, 0.6, red, 2)
            
        angle_text_string_neck = 'Neck angle: ' + str(int(neck_inclination))
        angle_text_string_torso = 'Torso angle: ' + str(int(torso_inclination))
        
        # Use user-defined thresholds
        if neck_inclination < neck_threshold and torso_inclination < torso_threshold:
            bad_frames = 0
            good_frames += 1
            posture_history.append(1)  # 1 for good posture
            
            cv2.putText(image, angle_text_string_neck, (10, 30), font, 0.6, light_green, 2)
            cv2.putText(image, angle_text_string_torso, (10, 60), font, 0.6, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
            
            # Draw posture lines
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 2)
            
            status_text.success(f"Good posture detected. Keep it up!")
            alert_text.empty()
            
            # Clear the benefits display when posture is good
            if benefits_displayed:
                posture_benefits_container.empty()
                benefits_displayed = False
                
        else:
            good_frames = 0
            bad_frames += 1
            posture_history.append(0)  # 0 for bad posture
            
            cv2.putText(image, angle_text_string_neck, (10, 30), font, 0.6, red, 2)
            cv2.putText(image, angle_text_string_torso, (10, 60), font, 0.6, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
            
            # Draw posture lines
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 2)
            
            status_text.error(f"Bad posture detected! Please adjust your position.")
            
            # Show posture benefits after detecting bad posture for 5 seconds
            bad_time = (1 / fps) * bad_frames
            if bad_time > 5 and not benefits_displayed:
                posture_benefits_container.markdown(
                    format_posture_benefits(),
                    unsafe_allow_html=True
                )
                benefits_displayed = True
        
        # Calculate times and update UI
        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames
        
        # Update posture stats
        good_time_display.metric("Good Posture Time", f"{round(good_time, 1)}s")
        bad_time_display.metric("Bad Posture Time", f"{round(bad_time, 1)}s")
        
        # Calculate posture quality as percentage
        if len(posture_history) > 0:
            quality = sum(posture_history) / len(posture_history) * 100
            posture_quality.metric("Posture Quality", f"{round(quality, 1)}%")
            
            # Limit history length to prevent memory issues
            if len(posture_history) > 300:  # Keep ~10 seconds at 30fps
                posture_history.pop(0)
        
        if good_time > 0:
            time_string_good = 'Good Posture: ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture: ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)
        
        # Send alert after user-defined time
        if bad_time > alert_time and not bad_posture_alert_sent and email_agent:
            try:
                prompt = f"""
Send an email to {user_email} with the subject 'Bad Posture Alert' and the body 'Hello,\n\nYour posture monitor has detected that you've been in a bad posture for {round(bad_time, 1)} seconds. Please take a moment to correct your sitting position.\n\nKey points to remember:\n- Keep your back straight\n- Position your screen at eye level\n- Keep shoulders relaxed\n\nBest regards,\nYour Posture Monitor'
"""
                email_agent.print_response(prompt)
                alert_text.warning(f"⚠️ Email alert sent to {user_email} for bad posture!")
                bad_posture_alert_sent = True
            except Exception as e:
                alert_text.error(f"Failed to send email alert: {str(e)}")
            
        if good_time > 0:
            bad_posture_alert_sent = False
            
        # Add timestamp to image
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(image, timestamp, (w - 150, h - 20), font, 0.7, white, 2)
            
        FRAME_WINDOW.image(image, channels="BGR")
        
        # Control the frame rate to reduce CPU usage
        time.sleep(0.01)
        
    if cap:
        cap.release()

if __name__ == "__main__":
    main()
