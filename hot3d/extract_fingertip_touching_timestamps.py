#!/usr/bin/env python3
"""
Script to extract timestamps where fingertips from either hand touch any object.

Touching is defined as when the 5 points on the fingertips (MANO hand landmarks 
indices 0,1,2,3,4) get close to the object pointcloud, parametrized by a threshold epsilon.
"""

import os
from pathlib import Path
import sys
import json
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
import pathlib
import tqdm

import trimesh

from data_loaders.loader_hand_poses import Handedness
from data_loaders.loader_object_library import ObjectLibrary, load_object_library
from data_loaders.ObjectPose3dProvider import TimeDomain, TimeQueryOptions
from dataset_api import Hot3dDataProvider
from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core.sophus import SE3  # @manual


@dataclass
class TouchEvent:
    """Data structure to store information about a fingertip touching event."""
    timestamp_ns: int
    handedness: str  # "left" or "right"
    hand_pose_matrix: Optional[List[List[float]]]  # 4x4 transformation matrix (None if wrist_pose is None)
    hand_joint_angles: List[float]
    hand_landmarks: List[List[float]]  # All 21 landmarks, each as [x, y, z]
    object_uid: str
    object_pose_matrix: List[List[float]]  # 4x4 transformation matrix
    camera_pose_matrix: Optional[List[List[float]]]  # 4x4 transformation matrix (None if device pose is not available)


def se3_to_matrix_list(se3: SE3) -> List[List[float]]:
    """Convert SE3 to a 4x4 matrix as a list of lists."""
    matrix = se3.to_matrix()
    return matrix.tolist()


def load_object_pointclouds(
    hot3d_data_provider: Hot3dDataProvider,
    num_samples: int = 10000,
) -> dict:
    """
    Load all object meshes and sample points from their surfaces to create pointclouds.
    
    Args:
        hot3d_data_provider: HOT3D data provider instance
        num_samples: Number of points to sample from each mesh surface (default: 10000)
    
    Returns:
        dict: Dictionary mapping object_uid to pointcloud vertices (numpy array of shape (N, 3))
    """
    object_library = hot3d_data_provider.object_library
    object_library_folderpath = object_library.asset_folder_name
    object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
    object_uids = object_pose_data_provider.object_uids_with_poses
    
    object_pointclouds = {}
    for object_uid in object_uids:
        object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
            object_library_folderpath=object_library_folderpath,
            object_id=object_uid,
        )
        
        if not os.path.exists(object_cad_asset_filepath):
            print(f"Warning: Object file not found: {object_cad_asset_filepath}")
            continue
            
        # Load the mesh
        scene = trimesh.load_mesh(
            object_cad_asset_filepath,
            process=True,
            merge_primitives=True,
            file_type="glb",
        )
        # Convert scene to a single mesh
        
        # Sample points from the mesh surface
        # This samples points uniformly from the surface of the mesh
        pointcloud, face_indices = trimesh.sample.sample_surface_even(scene, num_samples, seed=424242)
        object_pointclouds[object_uid] = pointcloud
    
    return object_pointclouds


def transform_pointcloud_to_world(
    pointcloud: np.ndarray,
    T_world_object: SE3,
) -> np.ndarray:
    """
    Transform pointcloud from object's local coordinate frame to world frame.
    
    Args:
        pointcloud: numpy array of shape (N, 3) in object's local frame
        T_world_object: SE3 transformation from object to world frame
    
    Returns:
        numpy array of shape (N, 3) in world frame
    """
    # Convert SE3 to 4x4 transformation matrix
    transform_matrix = T_world_object.to_matrix()
    
    # Convert pointcloud to homogeneous coordinates
    N = pointcloud.shape[0]
    points_homogeneous = np.hstack([pointcloud, np.ones((N, 1))])
    
    # Transform points
    points_world_homogeneous = (transform_matrix @ points_homogeneous.T).T
    
    # Return 3D coordinates
    return points_world_homogeneous[:, :3]


def check_fingertip_touching(
    fingertip_positions: np.ndarray,
    object_pointcloud_world: np.ndarray,
    epsilon: float,
) -> bool:
    """
    Check if any fingertip is within epsilon distance of any point in the object pointcloud.
    
    Args:
        fingertip_positions: numpy array of shape (5, 3) - positions of 5 fingertips in world frame
        object_pointcloud_world: numpy array of shape (N, 3) - object pointcloud in world frame
        epsilon: distance threshold in meters
    
    Returns:
        bool: True if any fingertip is within epsilon distance of any object point
    """
    if len(object_pointcloud_world) == 0:
        return False
    
    # Compute distances from each fingertip to all object points
    # fingertip_positions: (5, 3), object_pointcloud_world: (N, 3)
    # distances: (5, N)
    distances = np.linalg.norm(
        fingertip_positions[:, np.newaxis, :] - object_pointcloud_world[np.newaxis, :, :],
        axis=2
    )
    
    # Check if any fingertip is within epsilon distance of any object point
    min_distances = np.min(distances, axis=1)  # (5,) - min distance for each fingertip
    return bool(np.any(min_distances < epsilon))


def extract_fingertip_touching_timestamps(
    sequence_folder: str,
    object_library: ObjectLibrary,
    mano_hand_model: MANOHandModel,
    epsilon: float = 0.01,
    num_samples: int = 10000,
) -> List[TouchEvent]:
    """
    Extract timestamps where fingertips from either hand touch any object.
    
    Args:
        sequence_folder: Path to the HOT3D sequence folder (e.g., containing mano_hand_pose_trajectory.jsonl)
        object_library_folder: Path to the object library folder (containing instance.json and *.glb files)
        mano_hand_model_path: Optional path to MANO hand model directory (containing MANO_LEFT.pkl and MANO_RIGHT.pkl)
        epsilon: Distance threshold in meters for defining "touching" (default: 0.01m = 1cm)
        num_samples: Number of points to sample from each object mesh surface (default: 10000)
    
    Returns:
        List of TouchEvent objects containing detailed information about each touching event
    """
    
    # Initialize HOT3D data provider
    hot3d_data_provider = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        mano_hand_model=mano_hand_model,
        fail_on_missing_data=False,
    )
    
    # Get hand data provider (prefer MANO, fallback to UmeTrack)
    hand_data_provider = (
        hot3d_data_provider.mano_hand_data_provider
        if hot3d_data_provider.mano_hand_data_provider is not None
        else hot3d_data_provider.umetrack_hand_data_provider
    )
    
    if hand_data_provider is None:
        raise RuntimeError("No hand data provider available (neither MANO nor UmeTrack)")
    
    # Get object pose provider
    object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
    if object_pose_data_provider is None:
        raise RuntimeError("No object pose data provider available")
    
    # Get device pose provider (for camera pose)
    device_pose_data_provider = hot3d_data_provider.device_pose_data_provider
    
    # Load object pointclouds (in object's local coordinate frame)
    print("Loading object pointclouds...")
    object_pointclouds_local = load_object_pointclouds(hot3d_data_provider, num_samples=num_samples)
    print(f"Loaded {len(object_pointclouds_local)} object pointclouds")
    
    # Get all timestamps
    timestamps = hand_data_provider.timestamp_ns_list
    print(f"Processing {len(timestamps)} timestamps...")
    
    touch_events = []
    
    # Process each timestamp
    for i, timestamp_ns in enumerate(timestamps):
        if (i + 1) % 100 == 0:
            print(f"Processing timestamp {i+1}/{len(timestamps)}...")
        
        # Get hand poses at this timestamp
        hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=None,
        )
        
        if hand_poses_with_dt is None:
            continue
        
        hand_poses_collection = hand_poses_with_dt.pose3d_collection
        
        # Get object poses at this timestamp
        object_poses_with_dt = object_pose_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=None,
        )
        
        if object_poses_with_dt is None:
            continue
        
        object_poses_collection = object_poses_with_dt.pose3d_collection
        
        # Get device pose at this timestamp (for camera pose)
        camera_pose_matrix = None
        if device_pose_data_provider is not None:
            device_pose_with_dt = device_pose_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
                acceptable_time_delta=None,
            )
            if device_pose_with_dt is not None and device_pose_with_dt.pose3d.T_world_device is not None:
                camera_pose_matrix = se3_to_matrix_list(device_pose_with_dt.pose3d.T_world_device)
        
        # Check both left and right hands
        for handedness in [Handedness.Left, Handedness.Right]:
            if handedness not in hand_poses_collection.poses:
                continue
            
            hand_pose = hand_poses_collection.poses[handedness]
            
            # Get hand landmarks (21 landmarks, indices 0-4 are fingertips)
            hand_landmarks = hand_data_provider.get_hand_landmarks(hand_pose)
            
            if hand_landmarks is None:
                continue
            
            # Extract fingertip positions (indices 0, 1, 2, 3, 4)
            fingertip_positions = hand_landmarks[:5].numpy()  # Shape: (5, 3)
            
            # Get all hand landmarks (21 landmarks) for saving
            all_landmarks = hand_landmarks.numpy()  # Shape: (21, 3)
            
            # Check against each object
            for object_uid, object_pose3d in object_poses_collection.poses.items():
                if object_uid not in object_pointclouds_local:
                    continue
                
                # Transform object pointcloud to world coordinates
                T_world_object = object_pose3d.T_world_object
                object_pointcloud_world = transform_pointcloud_to_world(
                    object_pointclouds_local[object_uid],
                    T_world_object,
                )
                
                # Check if fingertips are touching the object
                if check_fingertip_touching(
                    fingertip_positions,
                    object_pointcloud_world,
                    epsilon,
                ):
                    # Create touch event with all details
                    touch_event = TouchEvent(
                        timestamp_ns=timestamp_ns,
                        handedness=hand_pose.handedness_label(),
                        hand_pose_matrix=se3_to_matrix_list(hand_pose.wrist_pose) if hand_pose.wrist_pose is not None else None,
                        hand_joint_angles=hand_pose.joint_angles,
                        hand_landmarks=all_landmarks.tolist(),
                        object_uid=object_uid,
                        object_pose_matrix=se3_to_matrix_list(T_world_object),
                        camera_pose_matrix=camera_pose_matrix,
                    )
                    touch_events.append(touch_event)
    
    return touch_events


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract timestamps where fingertips touch objects"
    )
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="Path to HOT3D sequence folder",
        default="/data/pickanything/hot3d/P0003_c701bd11",
    )
    parser.add_argument(
        "--object_library_folder",
        type=str,
        help="Path to object library folder",
        default="/data/pickanything/hot3d/assets",
    )
    parser.add_argument(
        "--mano_hand_model_path",
        type=str,
        help="Path to MANO hand model directory (optional)",
        default="/data/pickanything/mano_v1_2/models",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Distance threshold in meters for touching (default: 0.01m = 1cm)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of points to sample from each object mesh surface (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save touch events as JSON (optional, prints to stdout if not provided)",
    )
    
    
    # Load MANO hand model if provided

    args = parser.parse_args()
    mano_hand_model = MANOHandModel(args.mano_hand_model_path)
    # Load object library
    object_library = load_object_library(args.object_library_folder)
    
    all_folders = pathlib.Path(args.sequence_folder).glob("*")
    for folder in tqdm.tqdm(list[Path](all_folders)):
        if folder.is_dir():
            sequence_folder = folder.as_posix()

            try:
                # Extract touching events
                touch_events = extract_fingertip_touching_timestamps(
                    sequence_folder=sequence_folder,
                    object_library=object_library,
                    mano_hand_model=mano_hand_model,
                    epsilon=args.epsilon,
                    num_samples=args.num_samples,
                )
                output_filepath = os.path.join(sequence_folder, "touch_events.json")
                events_dict = [asdict(event) for event in touch_events]
                with open(output_filepath, "w") as f:
                    json.dump(events_dict, f, indent=2)
                print(f"\nTouch events saved to: {output_filepath}")

                # # Output results
                print(f"\nFound {len(touch_events)} fingertip touching events")
            except Exception as e:
                print(f"Error extracting touching events for {sequence_folder}: {e}")
                continue
            # if len(touch_events) > 0:
            #     print(f"First event: timestamp={touch_events[0].timestamp_ns}, "
            #         f"hand={touch_events[0].handedness}, object={touch_events[0].object_uid}")
            #     if len(touch_events) > 1:
            #         print(f"... and {len(touch_events) - 1} more events")

if __name__ == "__main__":
    main()

