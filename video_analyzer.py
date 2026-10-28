"""
Goal Probability Simulator - Video Analyzer
Uses YOLOv8 for real-time object detection on football/soccer footage
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_x: float
    center_y: float


@dataclass
class Ball:
    """Ball tracking data."""
    position: Tuple[float, float]
    trajectory: List[Tuple[float, float]]
    speed: float
    in_play: bool


@dataclass
class Player:
    """Player tracking data."""
    track_id: int
    position: Tuple[float, float]
    team: str  # 'home', 'away', 'referee', 'unknown'
    role: str  # 'attacker', 'defender', 'goalkeeper', 'other'
    trajectory: List[Tuple[float, float]]


@dataclass
class GoalFrame:
    """Goal frame detection."""
    detected: bool
    corners: List[Tuple[float, float]]
    center: Tuple[float, float]


class VideoAnalyzer:
    """
    Analyzes football/soccer video clips to extract
    information needed for xG calculation.
    
    Uses YOLOv8 for object detection:
    - Class 0: Person (players, referees)
    - Class 32: Sports ball (the ball)
    """
    
    # COCO class names for YOLOv8
    YOLO_CLASSES = {
        0: 'person',
        32: 'sports ball',
    }
    
    # Pitch dimensions (in meters, standard 105x68)
    PITCH_LENGTH = 105
    PITCH_WIDTH = 68
    
    # Goal position (standard: far end of pitch)
    GOAL_X = 105
    GOAL_Y_CENTER = 34
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initialize the video analyzer.
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt is nano, fastest)
                       Use yolov8s.pt for small, yolov8m.pt for medium
        """
        print(f"🚀 Initializing Video Analyzer...")
        print(f"   Model: {model_path}")
        
        # Check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Initialize tracking
        self.tracked_players: Dict[int, Player] = {}
        self.tracked_ball: Optional[Ball] = None
        self.goal_frame: Optional[GoalFrame] = None
        self.frame_count = 0
        
        # Pitch perspective matrix (will be calibrated)
        self.perspective_matrix = None
        
        print(f"✅ Video Analyzer ready!\n")
    
    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in a single frame using YOLOv8.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detected objects
        """
        # Run inference
        results = self.model(frame, verbose=False, conf=0.5)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Only process relevant classes (person and ball)
                if class_id not in self.YOLO_CLASSES:
                    continue
                
                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detection = DetectedObject(
                    class_id=class_id,
                    class_name=self.YOLO_CLASSES[class_id],
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2
                )
                detections.append(detection)
        
        return detections
    
    def estimate_pitch_coordinates(self, detections: List[DetectedObject], 
                                    frame_shape: Tuple[int, int]) -> Dict:
        """
        Estimate pitch coordinates from frame coordinates.
        Uses vanishing point and heuristics for soccer pitch.
        
        Args:
            detections: Detected objects
            frame_shape: (height, width) of frame
            
        Returns:
            Dictionary with coordinate mappings
        """
        frame_h, frame_w = frame_shape
        
        # Simple linear mapping (can be improved with homography)
        # Assuming goal is visible in upper portion of frame when shooting
        
        mappings = {
            'ball': None,
            'players': [],
            'goal_center_estimate': (frame_w * 0.5, frame_h * 0.15)
        }
        
        # Separate balls and players
        balls = [d for d in detections if d.class_name == 'sports ball']
        players = [d for d in detections if d.class_name == 'person']
        
        if balls:
            # Ball is typically small, estimate as percentage of frame
            ball = balls[0]  # Take first/closest ball
            mappings['ball'] = {
                'frame_x': ball.center_x,
                'frame_y': ball.center_y,
                'x_percent': ball.center_x / frame_w,
                'y_percent': ball.center_y / frame_h,
                'bbox': ball.bbox
            }
        
        for player in players:
            mappings['players'].append({
                'frame_x': player.center_x,
                'frame_y': player.center_y,
                'x_percent': player.center_x / frame_w,
                'y_percent': player.center_y / frame_h,
                'bbox': player.bbox,
                'confidence': player.confidence
            })
        
        return mappings
    
    def estimate_shot_zone(self, mappings: Dict) -> str:
        """
        Estimate which zone of the pitch the shot is taken from.
        
        Zones:
        - "penalty_spot": Inside 6-yard box center
        - "six_yard_box": Inside 6-yard box
        - "goal_area": Inside 18-yard box
        - "penalty_area": Inside 18-yard box
        - "edge_of_box": On 18-yard box line
        - "long_range": Outside 18-yard box
        - "very_long": 25+ meters out
        """
        if not mappings.get('ball'):
            return "unknown"
        
        # Convert frame position to rough pitch position
        # This is simplified - real implementation needs homography
        x_pct = mappings['ball']['x_percent']
        y_pct = mappings['ball']['y_percent']
        
        # In a typical forward-looking shot, x increases going toward goal
        # Frame bottom = near camera (player position)
        # Frame top = far camera (goal)
        
        # Estimate distance based on ball size in frame
        ball_bbox = mappings['ball'].get('bbox', (0, 0, 0, 0))
        ball_height = ball_bbox[3] - ball_bbox[1]
        avg_frame_height = 720  # Assume 720p for relative calculation
        
        # Ball size inversely proportional to distance
        relative_size = ball_height / avg_frame_height
        
        # Rough distance estimation (very simplified)
        if relative_size > 0.15:
            distance_meters = 3  # Very close
        elif relative_size > 0.10:
            distance_meters = 8  # Penalty spot
        elif relative_size > 0.06:
            distance_meters = 12  # Edge of 6-yard box
        elif relative_size > 0.04:
            distance_meters = 16  # Inside 18-yard box
        elif relative_size > 0.02:
            distance_meters = 22  # Edge of penalty area
        else:
            distance_meters = 30  # Long range
        
        # Classify zone
        if distance_meters <= 4:
            return "penalty_spot"
        elif distance_meters <= 8:
            return "six_yard_box"
        elif distance_meters <= 12:
            return "goal_area"
        elif distance_meters <= 16:
            return "penalty_area"
        elif distance_meters <= 22:
            return "edge_of_box"
        elif distance_meters <= 28:
            return "long_range"
        else:
            return "very_long_range"
    
    def count_defenders_in_path(self, mappings: Dict, shot_zone: str) -> int:
        """
        Estimate number of defenders between ball and goal.
        
        Uses player positions relative to ball and shot direction.
        """
        if not mappings.get('players'):
            return 0
        
        ball = mappings.get('ball')
        if not ball:
            return 0
        
        # In typical shot analysis, goal is "above" the ball in frame
        # Defenders between ball and goal would be above ball in frame
        # (assuming forward camera view)
        
        ball_y = ball['frame_y']
        defenders_count = 0
        
        for player in mappings['players']:
            player_y = player['frame_y']
            player_conf = player['confidence']
            
            # Player is between ball and goal if:
            # 1. Player is above ball in frame (y < ball_y)
            # 2. Player is reasonably confident detection
            # 3. Player is not too far to the side
            
            if (player_y < ball_y and 
                player_conf > 0.6 and
                abs(player['frame_x'] - ball['frame_x']) < ball['frame_x'] * 0.8):
                defenders_count += 1
        
        # Adjust based on shot zone (closer = more likely to have blocking defenders)
        zone_multipliers = {
            'penalty_spot': 0,
            'six_yard_box': 0.5,
            'goal_area': 1,
            'penalty_area': 1.5,
            'edge_of_box': 2,
            'long_range': 1,
            'very_long_range': 0,
            'unknown': 0
        }
        
        multiplier = zone_multipliers.get(shot_zone, 0)
        
        return min(int(defenders_count * multiplier), 6)  # Cap at 6
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame for goal probability calculation.
        
        Args:
            frame: BGR image
            
        Returns:
            Dictionary with analysis results
        """
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Estimate pitch coordinates
        mappings = self.estimate_pitch_coordinates(detections, frame.shape)
        
        # Determine shot zone
        shot_zone = self.estimate_shot_zone(mappings)
        
        # Count defenders
        defenders = self.count_defenders_in_path(mappings, shot_zone)
        
        # Count total players
        num_players = len(mappings.get('players', []))
        
        return {
            'detections': detections,
            'mappings': mappings,
            'shot_zone': shot_zone,
            'defenders_blocking': defenders,
            'total_players_detected': num_players,
            'has_ball': mappings.get('ball') is not None,
            'ball_size_relative': mappings.get('ball', {}).get('bbox', (0, 0, 0, 0)),
        }
    
    def analyze_video(self, video_path: str, max_frames: int = 100) -> Dict:
        """
        Analyze entire video and return comprehensive statistics.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze
            
        Returns:
            Dictionary with video analysis results
        """
        print(f"📹 Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Video: {total_frames} frames @ {fps:.1f} fps")
        print(f"   Analyzing up to {max_frames} frames...\n")
        
        frame_results = []
        frame_idx = 0
        analyzed = 0
        
        while analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every few frames to save time
            if frame_idx % max(1, total_frames // max_frames) == 0:
                result = self.analyze_frame(frame)
                result['frame_number'] = frame_idx
                frame_results.append(result)
                analyzed += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Aggregate results
        zones = [r['shot_zone'] for r in frame_results]
        defenders = [r['defenders_blocking'] for r in frame_results]
        
        return {
            'total_frames_analyzed': len(frame_results),
            'video_frames': total_frames,
            'fps': fps,
            'shot_zones': zones,
            'avg_defenders': np.mean(defenders) if defenders else 0,
            'max_defenders': max(defenders) if defenders else 0,
            'ball_detection_rate': sum(1 for r in frame_results if r['has_ball']) / len(frame_results) if frame_results else 0,
            'avg_players': np.mean([r['total_players_detected'] for r in frame_results]),
            'frame_results': frame_results,
            'most_common_zone': max(set(zones), default='unknown') if zones else 'unknown'
        }
    
    def visualize_detections(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw detection results on frame for visualization.
        
        Args:
            frame: Original BGR frame
            result: Analysis result from analyze_frame
            
        Returns:
            Frame with visualizations drawn
        """
        vis_frame = frame.copy()
        detections = result['detections']
        
        # Color mapping
        colors = {
            'person': (0, 255, 0),   # Green
            'sports ball': (255, 0, 0),  # Blue
        }
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw info overlay
        info_text = [
            f"Zone: {result['shot_zone']}",
            f"Defenders: {result['defenders_blocking']}",
            f"Players: {result['total_players_detected']}",
            f"Ball: {'Yes' if result['has_ball'] else 'No'}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_frame


# Demo function
def demo_analyzer():
    """Demonstrate the video analyzer capabilities."""
    print("=" * 60)
    print("GOAL PROBABILITY VIDEO ANALYZER - DEMO")
    print("=" * 60 + "\n")
    
    analyzer = VideoAnalyzer()
    
    # Simulate frame analysis
    print("📊 Simulating frame analysis...")
    
    # Create a mock analysis result
    mock_result = {
        'shot_zone': 'penalty_area',
        'defenders_blocking': 2,
        'total_players_detected': 14,
        'has_ball': True,
        'frame_number': 45
    }
    
    print(f"\n🔍 Frame Analysis Results:")
    print(f"   Shot Zone: {mock_result['shot_zone']}")
    print(f"   Defenders Blocking: {mock_result['defenders_blocking']}")
    print(f"   Players Detected: {mock_result['total_players_detected']}")
    print(f"   Ball Detected: {mock_result['has_ball']}")
    
    # Zone to xG mapping
    zone_xg_ranges = {
        'penalty_spot': (0.75, 0.95),
        'six_yard_box': (0.55, 0.75),
        'goal_area': (0.35, 0.55),
        'penalty_area': (0.20, 0.35),
        'edge_of_box': (0.10, 0.20),
        'long_range': (0.03, 0.10),
        'very_long_range': (0.01, 0.03),
    }
    
    xg_range = zone_xg_ranges.get(mock_result['shot_zone'], (0.05, 0.15))
    avg_xg = (xg_range[0] + xg_range[1]) / 2
    
    print(f"\n⚽ Goal Probability Estimate:")
    print(f"   xG Range: {xg_range[0]:.2f} - {xg_range[1]:.2f}")
    print(f"   Average xG: {avg_xg:.2f} ({avg_xg*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("\n💡 To analyze real video, use:")
    print("   analyzer.analyze_video('path/to/video.mp4')")


if __name__ == "__main__":
    demo_analyzer()
