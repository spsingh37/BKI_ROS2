import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class OdometryToPoseConverter(Node):
    def __init__(self):
        super().__init__('odometry_to_pose_converter')
        
        # Subscriber for Odometry messages
        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/wamv/sensors/position/ground_truth_odometry',  # Change this to your Odometry topic
            self.odometry_callback,
            10
        )
        
        # Publisher for PoseStamped messages
        self.pose_publisher = self.create_publisher(PoseStamped, '/gt_pose', 10)
        self.get_logger().info('Odometry to PoseStamped Converter Node Initialized')

    def odometry_callback(self, msg: Odometry):
        # Create and populate a PoseStamped message
        pose_stamped_msg = PoseStamped()
        pose_stamped_msg.header.stamp = msg.header.stamp  # Use timestamp from Odometry data
        pose_stamped_msg.header.frame_id = msg.header.frame_id  # Use same frame as odometry
        pose_stamped_msg.pose = msg.pose.pose  # Copy pose information from Odometry

        # Publish the PoseStamped message
        self.pose_publisher.publish(pose_stamped_msg)
        self.get_logger().info(f'Published PoseStamped: {pose_stamped_msg}')


def main(args=None):
    rclpy.init(args=args)
    node = OdometryToPoseConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node terminated by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
