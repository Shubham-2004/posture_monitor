import streamlit as st
import numpy as np
import time
import os
import re
import math
from dotenv import load_dotenv

# Determine if we're on Streamlit Cloud
def is_running_in_cloud():
    return os.environ.get("STREAMLIT_SERVER_HEADLESS", "") == "1"

# Conditionally import cv2 and mediapipe only if not in cloud
if not is_running_in_cloud():
    import cv2
    import mediapipe as mp
    CLOUD_MODE = False
else:
    CLOUD_MODE = True
    st.warning("Running in cloud mode with limited functionality. Camera and real-time posture monitoring will be disabled.")

# Safe import for AGNO tools
from agno.tools.email import EmailTools
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools

# Load .env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    st.error("GROQ API key not found. Please add it to your .env or environment variables.")

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
        f"<a href='{url}' target='_blank'>{url}</a><br><br>"
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

    # Check if we're in cloud mode - if so, run demo mode
    if CLOUD_MODE:
        st.warning("Running in cloud demo mode. Camera functionality is not available.")
        # Show static demo image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a sample skeleton on the image
        # Draw head
        center = (320, 100)
        radius = 30
        color = (200, 200, 200)
        thickness = 2
        # Using numpy to draw the circle
        for i in range(radius - thickness, radius + thickness):
            for angle in range(0, 360):
                x = int(center[0] + i * np.cos(np.radians(angle)))
                y = int(center[1] + i * np.sin(np.radians(angle)))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
                    
        # Draw body
        pt1 = (320, 130)
        pt2 = (320, 300)
        thickness = 2
        # Draw vertical line for body
        for i in range(-thickness, thickness):
            x1, y1 = pt1
            x2, y2 = pt2
            for t in np.linspace(0, 1, 100):
                x = int(x1 + t * (x2 - x1) + i)
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
                    
        # Draw arms
        pt1 = (250, 180)
        pt2 = (390, 180)
        pt3 = (320, 180)
        # Left arm
        for i in range(-thickness, thickness):
            for t in np.linspace(0, 1, 100):
                x = int(pt3[0] + t * (pt1[0] - pt3[0]))
                y = int(pt3[1] + t * (pt1[1] - pt3[1]) + i)
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
        # Right arm
        for i in range(-thickness, thickness):
            for t in np.linspace(0, 1, 100):
                x = int(pt3[0] + t * (pt2[0] - pt3[0]))
                y = int(pt3[1] + t * (pt2[1] - pt3[1]) + i)
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
                    
        # Draw legs
        pt1 = (270, 400)
        pt2 = (370, 400)
        pt3 = (320, 300)
        # Left leg
        for i in range(-thickness, thickness):
            for t in np.linspace(0, 1, 100):
                x = int(pt3[0] + t * (pt1[0] - pt3[0]) + i)
                y = int(pt3[1] + t * (pt1[1] - pt3[1]))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
        # Right leg
        for i in range(-thickness, thickness):
            for t in np.linspace(0, 1, 100):
                x = int(pt3[0] + t * (pt2[0] - pt3[0]) + i)
                y = int(pt3[1] + t * (pt2[1] - pt3[1]))
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
                    
        # Add text
        font_size = 0.7
        font_thickness = 2
        text = "DEMO MODE - CAMERA NOT AVAILABLE"
        text_size = font_size * len(text) * 7  # Approximate width
        text_x = int((image.shape[1] - text_size) / 2)
        # Add text using numpy
        for i in range(len(text)):
            char_x = text_x + i * int(font_size * 14)
            char_y = 30
            for dx in range(-font_thickness, font_thickness + 1):
                for dy in range(-font_thickness, font_thickness + 1):
                    x, y = char_x + dx, char_y + dy
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        image[y, x] = (255, 255, 255)
                        
        FRAME_WINDOW.image(image)
        
        # Show posture benefits
        posture_benefits_container.markdown(
            format_posture_benefits(),
            unsafe_allow_html=True
        )
        
        # Set some example metrics
        good_time_display.metric("Good Posture Time", "N/A")
        bad_time_display.metric("Bad Posture Time", "N/A")
        posture_quality.metric("Posture Quality", "N/A")
        
        st.info("This is a demonstration mode. For the full posture monitoring experience, please run this application locally.")
        st.stop()
        
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
