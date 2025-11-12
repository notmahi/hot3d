import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

# Imports for rerun visualization
try:
    from data_loaders.hand_common import LANDMARK_CONNECTIVITY
    from data_loaders.loader_object_library import ObjectLibrary, load_object_library
    from projectaria_tools.core.sophus import SE3
except ImportError:
    # These are only needed for rerun mode
    LANDMARK_CONNECTIVITY = None
    ObjectLibrary = None
    load_object_library = None
    SE3 = None

# --- 1. Data Structure Definition ---

@dataclass
class TouchEvent:
    """Data structure to store information about a fingertip touching event."""
    timestamp_ns: int
    handedness: str  # "left" or "right"
    hand_pose_matrix: Optional[List[List[float]]]  # 4x4 transformation matrix
    hand_joint_angles: List[float]
    hand_landmarks: List[List[float]]  # All 21 landmarks, each as [x, y, z]
    object_uid: str
    object_pose_matrix: List[List[float]]  # 4x4 transformation matrix

    # Helper properties to get numpy arrays
    @property
    def hand_pose_np(self) -> Optional[np.ndarray]:
        if self.hand_pose_matrix is None:
            return None
        return np.array(self.hand_pose_matrix)

    @property
    def object_pose_np(self) -> np.ndarray:
        return np.array(self.object_pose_matrix)

# --- 2. Helper Functions ---

def load_events_from_json(filepath: str) -> List[TouchEvent]:
    """Loads a list of TouchEvents from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        events = [TouchEvent(**event_data) for event_data in data]
        
        # CRITICAL: Ensure data is sorted by time
        events.sort(key=lambda e: e.timestamp_ns)
        return events
    except FileNotFoundError:
        print(f"Error: Input file '{filepath}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'.")
        return []

def pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """
    Calculates a weighted distance between two 4x4 transformation matrices.
    Combines translational and rotational (axis-angle) distance.
    """
    # Translational distance
    trans_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])

    # Rotational distance
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    
    # Rotation matrix for the difference
    R_diff = R1.T @ R2
    
    # Get angle from axis-angle representation
    trace = np.trace(R_diff)
    # np.clip to handle numerical inaccuracies
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

    # A common weighting is (1.0 * translational_meters) + (0.5 * rotational_radians)
    # For simplicity here, we add them. You can tune this.
    return trans_dist + angle

# --- 3. Core Grasp Analysis Logic ---

def group_events(events: List[TouchEvent]) -> Dict[str, List[TouchEvent]]:
    """Groups events into continuous sequences by (handedness, object_uid)."""
    sequences = {}
    if not events:
        return {}

    current_key = None
    current_sequence = []
    # Process right hand events first, then left hand events
    for handedness in ['right', 'left']:
        current_key = None
        current_sequence = []
        for event in events:
            if event.handedness != handedness:
                continue
            # We can't process events without a hand pose
            if event.hand_pose_matrix is None:
                if current_sequence: # End the current sequence
                    seq_id = f"{current_key[0]}_{current_key[1]}_{current_sequence[0].timestamp_ns}"
                    sequences[seq_id] = current_sequence
                current_key = None
                current_sequence = []
                continue

            key = (event.handedness, event.object_uid)

            if key != current_key:
                if current_sequence: # Store completed sequence
                    seq_id = f"{current_key[0]}_{current_key[1]}_{current_sequence[0].timestamp_ns}"
                    sequences[seq_id] = current_sequence

                current_key = key
                current_sequence = [event]
            else:
                current_sequence.append(event)

        # Don't forget the last sequence
        if current_sequence:
            seq_id = f"{current_key[0]}_{current_key[1]}_{current_sequence[0].timestamp_ns}"
            sequences[seq_id] = current_sequence

    return sequences

def find_grasp_segment(
    sequence: List[TouchEvent], 
    stability_thresh: float, 
    motion_thresh: float,
    min_grasp_steps: int
) -> Optional[Tuple[int, int]]:
    """
    Analyzes a sequence and finds the start/end index of the grasp 'hold' phase.
    Returns (start_index, end_index) or None if no grasp is found.
    """
    if len(sequence) < 2:
        return None

    # 1. Calculate relative and object poses
    relative_poses = []
    object_poses = []
    for event in sequence:
        hand_pose = event.hand_pose_np
        obj_pose = event.object_pose_np
        
        # This check should be redundant thanks to grouping, but it's safe
        if hand_pose is None: 
            return None # Can't analyze a sequence with missing hand poses

        relative_poses.append(np.linalg.inv(hand_pose) @ obj_pose)
        object_poses.append(obj_pose)

    if len(relative_poses) < 2:
        return None

    # 2. Calculate derivatives (pose changes)
    grasp_stability = []
    object_motion = []
    for i in range(len(relative_poses) - 1):
        grasp_stability.append(pose_distance(relative_poses[i], relative_poses[i+1]))
        object_motion.append(pose_distance(object_poses[i], object_poses[i+1]))

    print(f"mean grasp stability: {np.mean(grasp_stability)}")
    print(f"mean object motion: {np.mean(object_motion)}")
    # 3. Find the "hold" segment
    is_grasping = [
        (stability < stability_thresh) and (motion > motion_thresh)
        for stability, motion in zip(grasp_stability, object_motion)
    ]

    # Find the longest contiguous run of `True` in `is_grasping`
    longest_run_start = -1
    longest_run_len = 0
    current_run_start = -1
    current_run_len = 0

    for i, is_grasp in enumerate(is_grasping):
        if is_grasp:
            if current_run_len == 0:
                current_run_start = i
            current_run_len += 1
        else:
            if current_run_len > longest_run_len:
                longest_run_len = current_run_len
                longest_run_start = current_run_start
            current_run_len = 0
            current_run_start = -1
    
    if current_run_len > longest_run_len: # Check last run
        longest_run_len = current_run_len
        longest_run_start = current_run_start

    # 4. Check if the run is long enough
    if longest_run_len >= min_grasp_steps:
        # A grasp at index `i` in `is_grasping` compares events `i` and `i+1`.
        # A run from `i_start` to `i_start + len - 1` involves events
        # from `i_start` to `(i_start + len - 1) + 1`.
        grasp_start_index = longest_run_start
        grasp_end_index = longest_run_start + longest_run_len
        return (grasp_start_index, grasp_end_index)
    
    return None

# --- 4. Output Modes ---

def output_stdout(
    contact_sequences: Dict[str, List[TouchEvent]],
    grasp_results: Dict[str, Tuple[int, int]],
    stability_thresh: float,
    motion_thresh: float,
    min_grasp_steps: int
) -> None:
    """Output results to stdout (default behavior)."""
    print(f"Found {len(contact_sequences)} continuous contact sequences.")
    
    grasp_count = 0
    for seq_id, sequence in contact_sequences.items():
        print(f"\n--- Analyzing Sequence '{seq_id}' ({len(sequence)} events) ---")
        
        if seq_id in grasp_results:
            start_idx, end_idx = grasp_results[seq_id]
            grasp_count += 1
            
            # Segment the original sequence
            pre_grasp_events = sequence[:start_idx]
            grasp_events = sequence[start_idx : end_idx + 1]  # +1 for inclusive
            post_grasp_events = sequence[end_idx + 1 :]
            
            print(f"✅ GRASP DETECTED:")
            print(f"  Pre-Grasp: {len(pre_grasp_events)} events (Indices 0:{start_idx - 1})")
            print(f"  Grasp:     {len(grasp_events)} events (Indices {start_idx}:{end_idx})")
            print(f"  Post-Grasp: {len(post_grasp_events)} events (Indices {end_idx + 1}:end)")
        else:
            pass
    
    print(f"\n--- Summary ---")
    print(f"Total grasps found: {grasp_count}")

def output_json(
    contact_sequences: Dict[str, List[TouchEvent]],
    grasp_results: Dict[str, Tuple[int, int]],
    output_path: str
) -> None:
    """Output grasp events as JSON: Dict[str, List[TouchEvent]]."""
    grasp_dict: Dict[str, List[TouchEvent]] = {}
    
    for seq_id, sequence in contact_sequences.items():
        if seq_id in grasp_results:
            start_idx, end_idx = grasp_results[seq_id]
            grasp_events = sequence[start_idx : end_idx + 1]  # +1 for inclusive
            grasp_dict[seq_id] = grasp_events
    
    # Convert TouchEvent objects to dictionaries for JSON serialization
    json_dict = {
        seq_id: [asdict(event) for event in events]
        for seq_id, events in grasp_dict.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"JSON output written to {output_path}")
    print(f"Found {len(grasp_dict)} grasp sequences")

def matrix_to_se3(matrix: List[List[float]]) -> SE3:
    """Convert a 4x4 matrix (list of lists) to SE3."""
    if SE3 is None:
        raise ImportError("projectaria_tools is required for rerun output mode")
    
    matrix_np = np.array(matrix, dtype=np.float64)
    # Use SE3.from_matrix if available, otherwise convert via quaternion
    try:
        se3 = SE3.from_matrix(matrix_np)
        return se3
    except (AttributeError, TypeError):
        # Fallback: convert rotation matrix to quaternion
        rotation = matrix_np[:3, :3]
        translation = matrix_np[:3, 3]
        # Convert rotation matrix to quaternion
        # Using scipy or manual conversion
        try:
            from scipy.spatial.transform import Rotation
            rotation_obj = Rotation.from_matrix(rotation)
            quat = rotation_obj.as_quat()  # [x, y, z, w]
            
            # SE3.from_quat_and_translation expects (w, [x, y, z], translation)
            # Returns a tuple, we need the first element
            se3_result = SE3.from_quat_and_translation(
                float(quat[3]),  # w
                quat[:3],  # [x, y, z]
                translation
            )
            se3 = se3_result[0] if isinstance(se3_result, tuple) else se3_result
            return se3
        except ImportError:
            # Manual quaternion conversion if scipy not available
            trace = np.trace(rotation)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s=4*qw
                w = 0.25 * s
                x = (rotation[2, 1] - rotation[1, 2]) / s
                y = (rotation[0, 2] - rotation[2, 0]) / s
                z = (rotation[1, 0] - rotation[0, 1]) / s
            else:
                if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
                    s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                    w = (rotation[2, 1] - rotation[1, 2]) / s
                    x = 0.25 * s
                    y = (rotation[0, 1] + rotation[1, 0]) / s
                    z = (rotation[0, 2] + rotation[2, 0]) / s
                elif rotation[1, 1] > rotation[2, 2]:
                    s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                    w = (rotation[0, 2] - rotation[2, 0]) / s
                    x = (rotation[0, 1] + rotation[1, 0]) / s
                    y = 0.25 * s
                    z = (rotation[1, 2] + rotation[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                    w = (rotation[1, 0] - rotation[0, 1]) / s
                    x = (rotation[0, 2] + rotation[2, 0]) / s
                    y = (rotation[1, 2] + rotation[2, 1]) / s
                    z = 0.25 * s
            
            # SE3.from_quat_and_translation returns a tuple, we need the first element
            se3_result = SE3.from_quat_and_translation(
                float(w),
                np.array([x, y, z]),
                translation
            )
            se3 = se3_result[0] if isinstance(se3_result, tuple) else se3_result
            return se3

def output_rerun(
    contact_sequences: Dict[str, List[TouchEvent]],
    grasp_results: Dict[str, Tuple[int, int]],
    output_path: str,
    object_library_folder: str
) -> None:
    """Output visualization to rerun file showing trajectories with hand skeletons and object meshes."""
    try:
        import rerun as rr
        from projectaria_tools.utils.rerun_helpers import ToTransform3D
    except ImportError:
        raise ImportError(
            "rerun and projectaria_tools are required for rerun output mode. "
            "Please install them: pip install rerun-sdk projectaria-tools"
        )
    
    if LANDMARK_CONNECTIVITY is None or ObjectLibrary is None or load_object_library is None:
        raise ImportError(
            "data_loaders modules are required for rerun output mode. "
            "Please ensure data_loaders.hand_common and data_loaders.loader_object_library are available."
        )
    
    # Load object library
    object_library = load_object_library(object_library_folder)
    
    # Initialize rerun
    rr.init("Grasp Trajectories Visualization", spawn=False)
    if output_path:
        print(f"Saving .rrd file to {output_path}")
        rr.save(output_path)
    
    # Track all sequence IDs and their time ranges for visibility control
    sequence_ids = []
    sequence_time_ranges = {}  # seq_id -> (start_time, end_time)
    
    # First, collect all time ranges and prepare sequence info
    for seq_id, sequence in contact_sequences.items():
        if not sequence or sequence[0].hand_pose_matrix is None:
            continue
        timestamps = [e.timestamp_ns for e in sequence if e.hand_pose_matrix is not None]
        if timestamps:
            sequence_time_ranges[seq_id] = (min(timestamps), max(timestamps))
            sequence_ids.append(seq_id)
    
    # Sort sequences by start time for proper ordering
    sequence_ids.sort(key=lambda sid: sequence_time_ranges[sid][0])
    
    # Visualize each sequence
    for seq_id in sequence_ids:
        sequence = contact_sequences[seq_id]
        if not sequence or sequence[0].hand_pose_matrix is None:
            continue
        
        handedness = sequence[0].handedness
        object_uid = sequence[0].object_uid
        
        # Get object name from library
        object_name = object_library.object_id_to_name_dict.get(
            object_uid, f"object_{object_uid}"
        )
        object_name = f"{object_name}_{object_uid}"
        
        # Organize each sequence under its own unique path
        sequence_path = f"world/sequences/{seq_id}"
        hand_path = f"{sequence_path}/hand_{handedness}"
        object_path = f"{sequence_path}/object_{object_name}"
        
        # Get the time range for this sequence
        seq_start_time, seq_end_time = sequence_time_ranges[seq_id]
        
        # Clear all other sequences at the start of this sequence's time range
        # This ensures only one sequence is visible at a time
        rr.set_time_nanos("timestamp", seq_start_time)
        for other_seq_id in sequence_ids:
            if other_seq_id != seq_id:
                rr.log(f"world/sequences/{other_seq_id}", rr.Clear.recursive())
        
        # Get object mesh path and log it at the sequence start time (not timeless)
        # This way it will be cleared when the sequence ends
        object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
            object_library_folderpath=object_library.asset_folder_name,
            object_id=object_uid,
        )
        rr.log(
            object_path,
            rr.Asset3D(path=object_cad_asset_filepath),
        )

        # Collect poses and timestamps for trajectory visualization
        hand_positions = []
        object_positions = []
        
        # First pass: collect all positions for trajectory
        for i, event in enumerate(sequence):
            if event.hand_pose_matrix is None:
                continue
            
            # Log hand pose
            hand_pose_np = event.hand_pose_np
            hand_translation = hand_pose_np[:3, 3]
            hand_positions.append(hand_translation.tolist())
            
            # Log object pose
            obj_pose_np = event.object_pose_np
            object_translation = obj_pose_np[:3, 3]
            object_positions.append(object_translation.tolist())
        
        rr.set_time_nanos("timestamp", seq_start_time)
        
        # Second pass: log individual poses, hand skeletons, and object transforms at their timestamps
        for i, event in enumerate(sequence):
            if event.hand_pose_matrix is None:
                continue
            
            timestamp_ns = event.timestamp_ns
            
            # Set time for this event
            rr.set_time_nanos("timestamp", timestamp_ns)
            
            # Log object pose transform
            obj_pose_se3 = matrix_to_se3(event.object_pose_matrix)
            rr.log(
                object_path,
                ToTransform3D(obj_pose_se3, False),
            )
            
            # Log hand skeleton using landmarks
            landmarks = np.array(event.hand_landmarks)
            
            # Convert landmarks to connected lines using LANDMARK_CONNECTIVITY
            # This creates line strips for the skeletal representation
            points = [
                connections
                for connectivity in LANDMARK_CONNECTIVITY
                for connections in [
                    [landmarks[it].tolist() for it in connectivity]
                ]
            ]

            # Log skeletal representation as line strips
            rr.log(
                f"{hand_path}/joints",
                rr.LineStrips3D(points, radii=0.002),
            )
            
            # Optionally log landmarks as points for better visibility
            rr.log(
                f"{hand_path}/landmarks",
                rr.Points3D(landmarks, radii=0.005),
            )

        # Clear this sequence at the end of its time range (so it disappears when moving to next sequence)
        rr.set_time_nanos("timestamp", seq_end_time + 1)  # +1 to ensure it's after the last event
        rr.log(f"{sequence_path}", rr.Clear.recursive())

    # At the very beginning, clear all sequences so the first one appears cleanly
    if sequence_ids:
        first_seq_id = sequence_ids[0]
        first_seq_start_time = sequence_time_ranges[first_seq_id][0]
        rr.set_time_nanos("timestamp", first_seq_start_time - 1)
        for seq_id in sequence_ids:
            rr.log(f"world/sequences/{seq_id}", rr.Clear.recursive())
    
    print(f"Rerun visualization saved to {output_path}")
    print(f"Visualized {len(contact_sequences)} sequences")
    print(f"Each sequence is shown one at a time based on timestamps - scrub through time to see different sequences")

# --- 5. Main Execution ---

def main(INPUT_FILE: str, output_mode: str = "stdout", output_path: Optional[str] = None, object_library_folder: Optional[str] = None):
    # --- ⚠️ Configuration: These values require tuning! ---

    STABILITY_THRESHOLD = 0.05 # Max pose_distance for "stable" relative pose
    MOTION_THRESHOLD = 0.1    # Min pose_distance for "object motion"
    MIN_GRASP_STEPS = 3         # Min number of timesteps for a valid grasp

    # ----------------------------------------------------

    all_events = load_events_from_json(INPUT_FILE)
    if not all_events:
        return

    if output_mode == "stdout":
        print(f"Loaded {len(all_events)} total events.")
    
    contact_sequences = group_events(all_events)
    
    # Analyze all sequences and collect grasp results
    grasp_results: Dict[str, Tuple[int, int]] = {}
    
    for seq_id, sequence in contact_sequences.items():
        if output_mode == "stdout":
            print(f"\n--- Analyzing Sequence '{seq_id}' ({len(sequence)} events) ---")
        
        result = find_grasp_segment(
            sequence, 
            STABILITY_THRESHOLD, 
            MOTION_THRESHOLD,
            MIN_GRASP_STEPS
        )

        if result:
            start_idx, end_idx = result
            grasp_results[seq_id] = (start_idx, end_idx)
    
    print(grasp_results)
    # Output based on mode
    if output_mode == "stdout":
        output_stdout(
            contact_sequences,
            grasp_results,
            STABILITY_THRESHOLD,
            MOTION_THRESHOLD,
            MIN_GRASP_STEPS
        )
    elif output_mode == "json":
        if output_path is None:
            output_path = "grasp_events.json"
        output_json(contact_sequences, grasp_results, output_path)
    elif output_mode == "rerun":
        if output_path is None:
            output_path = "grasp_trajectories.rrd"
        if object_library_folder is None:
            raise ValueError(
                "object_library_folder is required for rerun output mode. "
                "Please provide --object-library-folder argument."
            )
        output_rerun(contact_sequences, grasp_results, output_path, object_library_folder)
    else:
        raise ValueError(f"Unknown output mode: {output_mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify grasp segments from fingertip touch events.")
    parser.add_argument(
        "--input",
        type=str,
        default="touch_events.json",
        help="Path to the input touch events JSON file (default: touch_events.json)"
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["stdout", "json", "rerun"],
        default="stdout",
        help="Output mode: 'stdout' for console output, 'json' for JSON file, 'rerun' for rerun visualization (default: stdout)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (required for json and rerun modes, optional with defaults: grasp_events.json or grasp_trajectories.rrd)"
    )
    parser.add_argument(
        "--object-library-folder",
        type=str,
        default="/data/pickanything/hot3d/assets",
        help="Path to object library folder (required for rerun output mode)"
    )
    args = parser.parse_args()
    main(args.input, args.output_mode, args.output, args.object_library_folder)