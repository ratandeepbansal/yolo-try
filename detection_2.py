from ultralytics import YOLO
import cv2
import os

def detect_household_items(video_path, output_path=None, model_name='yolov8n.pt', conf_threshold=0.25):
    """
    Detect household items in a video using YOLOv8
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (optional)
        model_name: YOLOv8 model variant (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        conf_threshold: Confidence threshold for detections (0-1)
    """
    
    # Load YOLOv8 model
    print(f"Loading {model_name} model...")
    model = YOLO(model_name)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
    print(f"Total frames: {total_frames}")
    
    # Setup video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output will be saved to: {output_path}")
    
    # Household item classes in COCO dataset (YOLOv8 default)
    household_classes = {
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        73: 'book',
        74: 'clock',
        75: 'vase',
        84: 'book',
    }
    
    # Note: COCO dataset doesn't have 'mug' or 'jar' as separate classes
    # 'cup' will detect mugs, 'bottle' or 'vase' might detect jars
    
    frame_count = 0
    
    print("\nProcessing video...")
    print("Press 'q' to quit early")
    print("\nNote: YOLOv8 COCO model will detect:")
    print("  - Mugs as 'cup'")
    print("  - Jars might be detected as 'bottle' or 'vase'")
    print("  - Bowls as 'bowl'\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("\nEnd of video reached")
            break
        
        frame_count += 1
        
        # Run YOLOv8 inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Annotate frame with detections
        annotated_frame = results[0].plot()
        
        # Count detections by class
        detections = results[0].boxes
        item_counts = {}
        
        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0])
                if cls_id in household_classes:
                    item_name = household_classes[cls_id]
                    item_counts[item_name] = item_counts.get(item_name, 0) + 1
        
        # Display counts on frame
        y_offset = 30
        if item_counts:
            for item_type, count in sorted(item_counts.items()):
                text = f"{item_type.capitalize()}: {count}"
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
        else:
            cv2.putText(annotated_frame, "No items detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame number
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        if out:
            out.write(annotated_frame)
        
        # Display frame
        cv2.imshow('YOLOv8 Household Items Detection', annotated_frame)
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%)")
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete! Processed {frame_count} frames")
    if output_path:
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "./C2.mp4"  # Change this to your video file path
    OUTPUT_PATH = "output_household_detected.mp4"  # Output video path (set to None to disable saving)
    MODEL = "yolov8n.pt"  # Using nano model for speed
    CONFIDENCE = 0.25  # Confidence threshold (0-1) - you can increase this to 0.4 for fewer false positives
    
    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found!")
        print("Please update VIDEO_PATH with your actual video file path")
    else:
        # Run detection
        detect_household_items(
            video_path=VIDEO_PATH,
            output_path=OUTPUT_PATH,
            model_name=MODEL,
            conf_threshold=CONFIDENCE
        )