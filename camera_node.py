import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Initialize ROS node
rospy.init_node('real_time_predictor')

# Define ROS image subscriber callback
def image_callback(img_msg):
    bridge = CvBridge()
    # Convert ROS Image message to OpenCV format
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    
    # 这里处理图像，提取行人的轨迹等
    processed_data = preprocess_data(cv_image)

    # 将数据传入模型进行预测
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(processed_data).float()
        predictions = model(x_tensor)
        
    # 可视化结果
    display_predictions(predictions)

# Subscribe to the image topic
rospy.Subscriber("/camera/image_raw", Image, image_callback)

# Spin to keep the program alive
rospy.spin()
