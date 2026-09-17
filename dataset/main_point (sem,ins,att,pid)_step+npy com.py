# Create point cloud for each shape
# Created by: Jiahao Zhao
# Date: 2025-08-19

import random
import os
import gc
import time
from tqdm import tqdm
from multiprocessing.pool import Pool

from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone

import Utils.occ_utils as occ_utils
import feature_creation

import numpy as np
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

# Import occwl for edge point cloud extraction
from occwl.edge import Edge as occwl_Edge
from occwl.uvgrid import ugrid

def extract_face_point_cloud(face, face_to_sem_inst_info, faces_list, face_idx, num_points_planar=None, num_points_curved=None):
    global FACE_POINTS_PER_SIDE_PLANAR, FACE_POINTS_PER_SIDE_CURVED
    if num_points_planar is None:
        num_points_planar = FACE_POINTS_PER_SIDE_PLANAR
    if num_points_curved is None:
        num_points_curved = FACE_POINTS_PER_SIDE_CURVED
    """
    Extract 3D point cloud data from face using correct OpenCascade methods for sampling
    Different sampling densities for planar vs curved surfaces to ensure adequate sampling for circular holes
    
    Args:
        face: TopoDS_Face object
        face_to_sem_inst_info: Mapping from face to semantic instance information
        faces_list: List of faces for getting face index
        num_points_planar: Number of sampling points for planar faces (default uses global parameter)
        num_points_curved: Number of sampling points for curved faces (default uses global parameter)
        
    Returns:
        numpy.ndarray: Array with shape (N, 9) containing:
        - points (x,y,z): 3D coordinates
        - normals (nx,ny,nz): Normal vectors
        - semantic: Semantic label (start from 1, stock face is 25)
        - instance: Instance index (start from 1)
        - attribute: Attribute label (0=planar, 1=curved, 2=straight edge, 3=curved edge, 4=boundary straight, 5=boundary curved)
    """
    try:
        # Import necessary modules
        from OCC.Core.BRepTools import breptools_UVBounds
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRep import BRep_Tool_Surface
        from OCC.Core.GeomLProp import GeomLProp_SLProps
        from OCC.Core.gp import gp_Dir
        from OCC.Core.TopAbs import TopAbs_REVERSED
        
        # Get UV bounds
        u_min, u_max, v_min, v_max = breptools_UVBounds(face)
        
        # Get surface adaptor
        surface_adaptor = BRepAdaptor_Surface(face)
        
        # Get surface geometry object
        surface = BRep_Tool_Surface(face)
        if surface is None:
            return None, None
        
        # Determine face type EARLY to decide sampling density
        from OCC.Core.GeomAbs import GeomAbs_Plane
        surface_type = surface_adaptor.GetType()
        is_planar = (surface_type == GeomAbs_Plane)
        
        # Choose appropriate sampling density based on surface type
        num_points = num_points_planar if is_planar else num_points_curved
            
        # Initialize result arrays
        result_points = []
        result_normals = []
        
        # Sample in UV parameter space with uniform distribution
        # Avoid sampling at exact boundaries to prevent seam line issues on cylindrical surfaces
        # Use slightly offset grid to ensure uniform distribution
        for i in range(num_points):
            # For uniform sampling, use center of each grid cell instead of grid points
            # This avoids concentration at boundaries/seams
            u = u_min + (u_max - u_min) * (i + 0.5) / num_points
            for j in range(num_points):
                v = v_min + (v_max - v_min) * (j + 0.5) / num_points
                
                try:
                    # Check if UV point is inside face boundary
                    uv_point = gp_Pnt2d(u, v)
                    classifier = BRepClass_FaceClassifier()
                    classifier.Perform(face, uv_point, 1e-6)
                    
                    state = classifier.State()
                    
                    # Only process points inside or on face boundary
                    if state == TopAbs_IN or state == TopAbs_ON:
                        # Calculate 3D point coordinates
                        point_3d = surface_adaptor.Value(u, v)
                        
                        # Calculate normal vector
                        try:
                            sl_props = GeomLProp_SLProps(surface, u, v, 1, 0.01)
                            normal = sl_props.Normal()
                            if face.Orientation() == TopAbs_REVERSED:
                                normal.Reverse()
                        except:
                            # If normal calculation fails, use default value
                            normal = gp_Dir(0.0, 0.0, 1.0)
                        
                        result_points.append([point_3d.X(), point_3d.Y(), point_3d.Z()])
                        result_normals.append([normal.X(), normal.Y(), normal.Z()])
                        
                except Exception as e:
                    # If calculation fails, skip this point
                    continue
        
        if len(result_points) == 0:
            return None, None
            
        # Convert to numpy arrays
        points = np.array(result_points, dtype=np.float32)
        normals = np.array(result_normals, dtype=np.float32)
        
        # Get face semantic instance information
        info = face_to_sem_inst_info.get(face, None)
        if info:
            semantic_idx = info['semantic_index']
            instance_idx = info['instance_index']
        else:
            # Default to stock label: semantic 25, instance 1
            semantic_idx = 25
            instance_idx = 1
        
        # Assign same label information to all points
        num_points_actual = points.shape[0]
        
        # Create semantic and instance arrays (float32 for consistency with point cloud array)
        sem_array = np.full((num_points_actual, 1), float(semantic_idx), dtype=np.float32)
        ins_array = np.full((num_points_actual, 1), float(instance_idx), dtype=np.float32)
        
        # Determine face type: planar (att=0) or curved (att=1)
        # Note: is_planar was already computed earlier in the function
        if is_planar:
            att_value = 0.0  # Planar face
        else:
            att_value = 1.0  # Curved face (cylinder, cone, sphere, torus, etc.)
        
        attribute_array = np.full((num_points_actual, 1), att_value, dtype=np.float32)
        
        # Concatenate point coordinates, normal vectors and label information to form complete point cloud data
        # Output format: [x, y, z, nx, ny, nz, semantic, instance, attribute]
        point_cloud = np.concatenate([points, normals, sem_array, ins_array, attribute_array], axis=1)

        # Create face index array for mapping points back to faces
        face_indices = np.full((num_points_actual,), face_idx, dtype=np.int32)
        
        return point_cloud, face_indices
        
    except Exception as e:
        return None, None

def extract_edge_point_cloud(edge, shape, face_to_sem_inst_info, faces_list, edge_idx, num_points=None):
    global EDGE_POINTS_PER_EDGE
    if num_points is None:
        num_points = EDGE_POINTS_PER_EDGE
    """
    Extract point cloud data from edge and set sem_ins value based on semantic instance labels of connected faces
    
    Args:
        edge: TopoDS_Edge object
        shape: 3D shape object
        face_to_sem_inst_info: Mapping from face to semantic instance information
        faces_list: List of faces for getting face index
        num_points: Number of sampling points
        
    Returns:
        numpy.ndarray: Array with shape (N, 9) containing:
        - points (x,y,z): 3D coordinates
        - normals (nx,ny,nz): Tangent vectors for edges
        - semantic: Semantic label (all edges inherit face semantic, 1-25)
        - instance: Instance index (internal edge inherits face instance, boundary edge keeps original)
        - attribute: Attribute label (2=straight edge, 3=curved edge, 4=boundary straight, 5=boundary curved)
    """
    try:
        # Convert TopoDS_Edge to occwl.Edge object
        occwl_edge = occwl_Edge(edge)
        
        # Before point cloud sampling, determine semantic instance labels of two faces connected by the edge
        # Use OpenCASCADE topological relationships to directly find faces connected by the edge
        edge_faces = []
        
        # Iterate through all faces to find faces containing this edge
        # Note: A seam edge on a cylindrical surface may appear multiple times in the same face
        for face_idx, face in enumerate(faces_list):
            try:
                # Get all edges of the face
                face_edges = occ_utils.list_edge(face)
                # Check if edge belongs to this face (using IsSame comparison)
                edge_count_in_face = 0
                for face_edge in face_edges:
                    if face_edge.IsSame(edge):
                        edge_count_in_face += 1
                
                # If edge appears in this face, add it to edge_faces
                # For seam edges, it may appear twice (both ends of the seam)
                if edge_count_in_face > 0:
                    edge_faces.append((face_idx, face, edge_count_in_face))
            except Exception as e:
                continue
        
        # If faces connected by the edge are found, compare their semantic instance labels
        if len(edge_faces) >= 2:
            # Get semantic instance label of first face
            face1_idx, face1, edge_count1 = edge_faces[0]
            face1_sem_inst = face_to_sem_inst_info.get(face1, None)  # Use face object as key
            
            # Get semantic instance label of second face
            face2_idx, face2, edge_count2 = edge_faces[1]
            face2_sem_inst = face_to_sem_inst_info.get(face2, None)  # Use face object as key
            
            # If direct lookup fails, try using IsSame to find matching faces
            if face1_sem_inst is None:
                for face_obj, sem_inst in face_to_sem_inst_info.items():
                    if face_obj.IsSame(face1):
                        face1_sem_inst = sem_inst
                        break
            
            if face2_sem_inst is None:
                for face_obj, sem_inst in face_to_sem_inst_info.items():
                    if face_obj.IsSame(face2):
                        face2_sem_inst = sem_inst
                        break
            
            # CRITICAL CHECK: If both faces are actually the SAME face (self-intersection edge)
            # This happens on cylindrical surfaces where the seam/mother line creates a self-intersection
            # Such edges should be skipped entirely (no point cloud generated)
            # Multiple detection methods:
            # 1. Same face index (face1_idx == face2_idx)
            # 2. IsSame comparison
            # 3. Edge appears multiple times in a single face (edge_count > 1)
            if face1_idx == face2_idx or face1.IsSame(face2) or edge_count1 > 1 or edge_count2 > 1:
                # Same face - this is a seam edge (e.g., cylinder mother line)
                # Skip this edge entirely - do not generate point cloud
                return None, None
            
            # Compare semantic instance labels of two faces
            elif face1_sem_inst is not None and face2_sem_inst is not None:
                # Extract semantic instance labels for comparison
                face1_label = face1_sem_inst.get('letter_label', '')
                face2_label = face2_sem_inst.get('letter_label', '')
                face1_semantic = face1_sem_inst.get('semantic_index', 0)
                face2_semantic = face2_sem_inst.get('semantic_index', 0)
                
                # FINAL LOGIC: Semantic Instance Boundary Detection
                # Key insight: Bd维度标记语义实例边界，但边界点仍保留其sem标签
                # 这样在可视化sem时能看到完整的加工特征（包括边界），而单独可视化Bd时能看到语义实例边界
                
                # Check if both faces belong to the same semantic instance
                same_semantic_instance = (face1_semantic == face2_semantic and 
                                         face1_sem_inst.get('instance_index', 1) == face2_sem_inst.get('instance_index', 1))
                
                if same_semantic_instance:
                    # Same semantic instance - this edge connects two faces of the same feature
                    # Examples: circular edge between cylinder side and end cap of a hole
                    sem_ins_encoded = face1_sem_inst.get('semantic_index', 1) * 100 + face1_sem_inst.get('instance_index', 1)
                    sem_ins_value = float(sem_ins_encoded)
                    # If both faces are STOCK (semantic=25), this is an EXTERNAL edge (model surface)
                    # If both faces are FEATURES (semantic!=25), this is an INTERNAL edge (inside feature)
                    if face1_semantic == 25:
                        attribute_value = 3  # Stock surface edge - EXTERNAL (att=3)
                    else:
                        attribute_value = 2  # Feature internal edge - INTERNAL (att=2)
                    
                elif face1_semantic != 25 and face2_semantic == 25:
                    # Feature vs Stock - this IS a semantic instance boundary
                    # att=3, but keep feature's semantic label in sem dimension
                    sem_ins_encoded = face1_semantic * 100 + face1_sem_inst.get('instance_index', 1)
                    sem_ins_value = float(sem_ins_encoded)
                    attribute_value = 3  # Boundary edge (att=3)
                    
                elif face2_semantic != 25 and face1_semantic == 25:
                    # Stock vs Feature - this IS a semantic instance boundary
                    # att=3, but keep feature's semantic label in sem dimension
                    sem_ins_encoded = face2_semantic * 100 + face2_sem_inst.get('instance_index', 1)
                    sem_ins_value = float(sem_ins_encoded)
                    attribute_value = 3  # Boundary edge (att=3)
                    
                elif face1_semantic != 25 and face2_semantic != 25:
                    # Both are features (different semantic instances)
                    # THIS is also a semantic instance boundary - att=3
                    # Priority strategy for circular/cylindrical features:
                    # When a circular feature intersects another feature, prioritize keeping the circular feature's semantic label
                    # This ensures circular edges maintain their correct semantic classification
                    # Circular feature semantic indices (feature_index + 1):
                    # - Through hole: index=1 -> semantic=2
                    # - Circular through slot: index=7 -> semantic=8
                    # - O-ring: index=11 -> semantic=12
                    # - Blind hole: index=12 -> semantic=13
                    # - Circular end pocket: index=16 -> semantic=17
                    # - V circular end blind slot: index=18 -> semantic=19
                    # - H circular end blind slot: index=19 -> semantic=20
                    # - Circular blind step: index=21 -> semantic=22
                    hole_semantics = {2, 8, 12, 13, 17, 19, 20, 22}  # All circular/cylindrical features
                    
                    # Check if either face is a hole feature
                    face1_is_hole = face1_semantic in hole_semantics
                    face2_is_hole = face2_semantic in hole_semantics
                    
                    if face1_is_hole and not face2_is_hole:
                        # face1 is a hole, face2 is not - prioritize hole
                        sem_ins_encoded = face1_semantic * 100 + face1_sem_inst.get('instance_index', 1)
                        sem_ins_value = float(sem_ins_encoded)
                    elif face2_is_hole and not face1_is_hole:
                        # face2 is a hole, face1 is not - prioritize hole
                        sem_ins_encoded = face2_semantic * 100 + face2_sem_inst.get('instance_index', 1)
                        sem_ins_value = float(sem_ins_encoded)
                    else:
                        # Either both are holes or neither is a hole - choose smaller semantic index
                        if face1_semantic <= face2_semantic:
                            sem_ins_encoded = face1_semantic * 100 + face1_sem_inst.get('instance_index', 1)
                            sem_ins_value = float(sem_ins_encoded)
                        else:
                            sem_ins_encoded = face2_semantic * 100 + face2_sem_inst.get('instance_index', 1)
                            sem_ins_value = float(sem_ins_encoded)
                    attribute_value = 3  # Boundary edge (att=3)
                    
                else:
                    # Both are stock - edges on stock surface are EXTERNAL
                    sem_ins_encoded = face1_semantic * 100 + face1_sem_inst.get('instance_index', 1)
                    sem_ins_value = float(sem_ins_encoded)
                    attribute_value = 3  # Stock surface edge - EXTERNAL (att=3)
            else:
                # If semantic instance labels cannot be obtained, use first face's semantic and mark as boundary
                sem_ins_encoded = face1_semantic * 100 + face1_sem_inst.get('instance_index', 1) if face1_sem_inst else 2501
                sem_ins_value = float(sem_ins_encoded)
                attribute_value = 3  # Boundary edge (att=3)
        # Special case: edge appears only once in edge_faces but might be a seam edge
        elif len(edge_faces) == 1:
            face_idx, face, edge_count = edge_faces[0]
            
            # If edge appears multiple times in the SAME face, it's a seam edge
            # Skip this edge entirely - do not generate point cloud
            if edge_count > 1:
                return None, None
            else:
                # Edge appears only once - it's an EXTERNAL/FREE edge (att=3)
                # This is the outer boundary of the model
                face_sem_inst = face_to_sem_inst_info.get(face, None)
                if face_sem_inst is None:
                    for face_obj, sem_inst in face_to_sem_inst_info.items():
                        if face_obj.IsSame(face):
                            face_sem_inst = sem_inst
                            break
                
                if face_sem_inst is not None:
                    # Single face edge - inherit semantic from the face
                    # Mark as OUTER BOUNDARY (att=3)
                    sem_ins_encoded = face_sem_inst.get('semantic_index', 1) * 100 + face_sem_inst.get('instance_index', 1)
                    sem_ins_value = float(sem_ins_encoded)
                    attribute_value = 3  # External edge is outer boundary (att=3)
                else:
                    # No semantic info - default to stock semantic, mark as boundary
                    sem_ins_value = 2501.0  # Stock semantic 25, instance 1
                    attribute_value = 3  # Boundary edge (att=3)
        else:
            # No faces connected - external boundary, use stock semantic
            sem_ins_value = 2501.0  # Stock semantic 25, instance 1
            attribute_value = 3  # Boundary edge (att=3)
        
        # Use ugrid for point sampling
        points = ugrid(occwl_edge, num_points, method="point")
        
        if points is None:
            return None, None
            
        # Ensure points are float32 type
        points = points.astype(np.float32)
            
        # Calculate tangent vectors (simple difference method)
        tangents = np.zeros_like(points)
        if len(points) > 1:
            # Forward difference
            tangents[:-1] = points[1:] - points[:-1]
            # Last point uses backward difference
            tangents[-1] = points[-1] - points[-2]
            
            # Normalize tangent vectors
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            tangents = tangents / norms
        
        # Get edge type (straight or curved edge)
        edge_type = occ_utils.type_edge(edge)
        is_straight_edge = (edge_type == 'line')  # 'line' means straight edge
        
        # Adjust attribute_value based on edge geometry (straight vs curved)
        # Current attribute_value is either 2/3 (internal) or will be set later
        # We need to map: internal straight=2, internal curved=3, boundary straight=4, boundary curved=5
        
        # Determine final attribute value based on boundary status and edge geometry
        if attribute_value == 2:
            # Internal edge: 2 for straight, 3 for curved
            final_att_value = 2 if is_straight_edge else 3
        elif attribute_value == 3:
            # Boundary edge: 4 for straight, 5 for curved
            final_att_value = 4 if is_straight_edge else 5
        else:
            # Fallback (shouldn't happen, but just in case)
            final_att_value = attribute_value
        
        # Assign semantic and instance labels to all points (based on connected faces)
        # Extract semantic and instance from the encoded value
        sem_value = float(int(sem_ins_value) // 100)
        ins_value = float(int(sem_ins_value) % 100)
        
        semantic_labels = np.full((points.shape[0], 1), sem_value, dtype=np.float32)
        instance_labels = np.full((points.shape[0], 1), ins_value, dtype=np.float32)
        
        # Attribute label: use final_att_value (2/3/4/5) determined above
        attribute_values = np.full((points.shape[0], 1), float(final_att_value), dtype=np.float32)
        
        # Concatenate all information to form complete edge point cloud data
        # Output format: [x, y, z, nx, ny, nz, semantic, instance, attribute]
        edge_point_cloud = np.concatenate([points, tangents, semantic_labels, instance_labels, attribute_values], axis=1)

        # Create edge index array for mapping points back to edges (negative to distinguish from faces)
        edge_indices = np.full((points.shape[0],), -(edge_idx + 1), dtype=np.int32)
        
        return edge_point_cloud, edge_indices
        
    except Exception as e:
        # On failure for a single edge, return (None, None) so that the caller can skip this edge
        return None, None

def extract_all_edges_point_cloud(shape, face_to_sem_inst_info, faces_list, num_points=None):
    global EDGE_POINTS_PER_EDGE
    if num_points is None:
        num_points = EDGE_POINTS_PER_EDGE
    """
    Extract point cloud data from all edges in the shape
    
    Args:
        shape: 3D shape object
        face_to_sem_inst_info: Mapping from face to semantic instance information
        faces_list: List of faces for getting face index
        num_points: Number of sampling points per edge
        
    Returns:
        numpy.ndarray: Point cloud data from all edges
    """
    try:
        # Get all edges
        edges_list = occ_utils.list_edge(shape)
        
        all_edge_point_clouds = []
        all_edge_indices = []
        
        for edge_idx, edge in enumerate(edges_list):
            edge_point_cloud, edge_indices = extract_edge_point_cloud(
                edge, shape, face_to_sem_inst_info, faces_list, edge_idx, num_points
            )
            if edge_point_cloud is not None:
                all_edge_point_clouds.append(edge_point_cloud)
                all_edge_indices.append(edge_indices)
        
        if all_edge_point_clouds:
            # Merge point clouds from all edges
            combined_edge_point_cloud = np.vstack(all_edge_point_clouds)
            combined_edge_indices = np.concatenate(all_edge_indices)
            return combined_edge_point_cloud, combined_edge_indices
        else:
            return np.empty((0, 9), dtype=np.float32), np.empty((0,), dtype=np.int32)
            
    except Exception as e:
        return np.empty((0, 9), dtype=np.float32), np.empty((0,), dtype=np.int32)

def save_point_cloud_npy(shape, npy_path, face_to_sem_inst_info, faces_list, num_points_planar=None, num_points_curved=None):
    global FACE_POINTS_PER_SIDE_PLANAR, FACE_POINTS_PER_SIDE_CURVED
    if num_points_planar is None:
        num_points_planar = FACE_POINTS_PER_SIDE_PLANAR
    if num_points_curved is None:
        num_points_curved = FACE_POINTS_PER_SIDE_CURVED
    """
    Convert shape to point cloud and save as .npy file, including face and edge point clouds.
    Mapping from point -> face/edge is stored directly in the structured npy as field 'pid'.
    Supports different sampling densities for planar vs curved surfaces.
    
    Args:
        shape: 3D shape object
        npy_path: Save path
        face_to_sem_inst_info: Mapping from face to semantic instance information
        faces_list: List of faces for getting face index
        num_points_planar: Number of sampling points for planar faces
        num_points_curved: Number of sampling points for curved faces
    """
    all_point_clouds = []
    all_indices = []  # Store face/edge indices for each point
    
    # 1. Extract face point clouds
    for face_idx, face in enumerate(faces_list):
        point_cloud, face_indices = extract_face_point_cloud(
            face, face_to_sem_inst_info, faces_list, face_idx, num_points_planar, num_points_curved
        )
        if point_cloud is not None:
            all_point_clouds.append(point_cloud)
            all_indices.append(face_indices)
    
    # 2. Extract edge point clouds
    edge_point_cloud, edge_indices = extract_all_edges_point_cloud(shape, face_to_sem_inst_info, faces_list)
    if edge_point_cloud is not None and len(edge_point_cloud) > 0:
        all_point_clouds.append(edge_point_cloud)
        all_indices.append(edge_indices)
    
    if all_point_clouds:
        # Merge all point clouds (faces and edges)
        combined_point_cloud = np.vstack(all_point_clouds)
        combined_indices = np.concatenate(all_indices)

        # Save as structured .npy with per-column dtypes:
        # - float32 for coords/normals
        # - uint8 for sem/ins/att
        # - int32 for pid (point -> face/edge index)
        struct_dtype = np.dtype([
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("nx", np.float32), ("ny", np.float32), ("nz", np.float32),
            ("sem", np.uint8), ("ins", np.uint8), ("att", np.uint8),
            ("pid", np.int32),  # new: per-point primitive id (face>=0, edge<0)
        ])
        struct_array = np.empty(combined_point_cloud.shape[0], dtype=struct_dtype)
        struct_array["x"] = combined_point_cloud[:, 0].astype(np.float32, copy=False)
        struct_array["y"] = combined_point_cloud[:, 1].astype(np.float32, copy=False)
        struct_array["z"] = combined_point_cloud[:, 2].astype(np.float32, copy=False)
        struct_array["nx"] = combined_point_cloud[:, 3].astype(np.float32, copy=False)
        struct_array["ny"] = combined_point_cloud[:, 4].astype(np.float32, copy=False)
        struct_array["nz"] = combined_point_cloud[:, 5].astype(np.float32, copy=False)
        struct_array["sem"] = np.clip(combined_point_cloud[:, 6], 0, 255).astype(np.uint8, copy=False)
        struct_array["ins"] = np.clip(combined_point_cloud[:, 7], 0, 255).astype(np.uint8, copy=False)
        struct_array["att"] = np.clip(combined_point_cloud[:, 8], 0, 5).astype(np.uint8, copy=False)
        struct_array["pid"] = combined_indices.astype(np.int32, copy=False)

        np.save(npy_path, struct_array)

from collections import defaultdict

def save_label(shape_name, pathname, seg_label, relation_matrix, bottom_label):
    import json
    """
    Export label data to a json file
    """
    data = [
        [shape_name, {'seg': seg_label, 'inst': relation_matrix, 'bottom': bottom_label}]
    ]
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def semantic_instance_labels(inst_map, label_map):
    # key: semantic label, value: current instance counter for this semantic label
    sem_count = defaultdict(int)
    # Result mapping: face object -> composite label string, e.g., "A_1"
    face_sem_inst_label = {}
    # Result mapping: face object -> dictionary containing numeric index and semantic name
    face_sem_inst_info = {}

    for inst_faces in inst_map:
        if len(inst_faces) == 0:
            continue
        # Take semantic label of first face in instance
        sem_label = label_map[inst_faces[0]]
        sem_count[sem_label] += 1
        inst_idx = sem_count[sem_label]
       
        sem_char = chr(ord('A') + (sem_label % 26))

        for face in inst_faces:
            face_sem_inst_label[face] = f"{sem_char}_{inst_idx}"
            # Also save numeric index and semantic name information
            # Semantic index +1 to avoid semantic 0 showing as 0.0 in float32
            face_sem_inst_info[face] = {
                'letter_label': f"{sem_char}_{inst_idx}",
                'semantic_index': sem_label + 1,  # Semantic index +1
                'instance_index': inst_idx
            }

    return face_sem_inst_label, face_sem_inst_info

def generate_shape(args):
    dataset_dir, combo = args
    f_name, combination = combo

    num_try = 0
    while True:
        num_try += 1

        if num_try > 3:
            break

        try:
            shape, labels = feature_creation.shape_from_directive(combination)
        except Exception as e:
            continue

        if shape is None:
            continue

        if not isinstance(shape, (TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)):
            continue

        # unpack labels
        seg_map, inst_map, bottom_map = labels

        faces_list = occ_utils.list_face(shape)
        if len(faces_list) == 0:
            continue

        seg_label = feature_creation.get_segmentaion_label(faces_list, seg_map)
        if len(seg_label) != len(faces_list):
            continue

        relation_matrix = feature_creation.get_instance_label(faces_list, len(seg_map), inst_map)
        if len(relation_matrix) != len(faces_list):
            continue

        # Create map between face id and bottom identification label
        bottom_label = feature_creation.get_segmentaion_label(faces_list, bottom_map)
        if len(bottom_label) != len(faces_list):
            continue

        # 🔹 Call semantic_instance_labels to generate face to "semantic_instance" mapping
        face_to_sem_inst, face_to_sem_inst_info = semantic_instance_labels(inst_map, seg_map)

        # Assign special semantic instance labels to stock faces
        # Stock face semantic label is 24, we assign it a special instance label
        for face in faces_list:
            if face not in face_to_sem_inst_info:
                # This is a stock face, assign it a special label
                face_to_sem_inst_info[face] = {
                    'letter_label': "Z_1",  # Use Z for stock to avoid conflict with A-Z shape labels
                    'semantic_index': 25,   # 25 corresponds to stock (24+1)
                    'instance_index': 1     # Stock faces typically have only one instance
                }
                # Note: This will result in encoding 2501 (25*100 + 1) for stock faces

        # Save data - save point cloud .npy file (with pid), corresponding STEP model, and label JSON
        shape_name = str(f_name)
        npy_path = os.path.join(dataset_dir, 'npys')
        step_path = os.path.join(dataset_dir, 'steps')
        label_path = os.path.join(dataset_dir, 'labels')
        os.makedirs(npy_path, exist_ok=True)
        os.makedirs(step_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)
        npy_file = os.path.join(npy_path, shape_name + '.npy')
        step_file = os.path.join(step_path, shape_name + '.step')
        label_file = os.path.join(label_path, shape_name + '.json')

        try:
            # Save point cloud npy (with pid in-struct mapping); no external mapping.npy is written
            save_point_cloud_npy(shape, npy_file, face_to_sem_inst_info, faces_list)

            # Save STEP CAD model with the same base filename
            step_writer = STEPControl_Writer()
            step_writer.Transfer(shape, STEPControl_AsIs)
            status = step_writer.Write(step_file)
            if status != IFSelect_RetDone:
                print(f"Warning: STEP file write failed for {shape_name}")
                continue

            # Save label JSON with seg/inst/bottom labels
            save_label(shape_name, label_file, seg_label, relation_matrix, bottom_label)
        except Exception as e:
            continue

        break

    return

def initializer():
    import signal
    """
    Ignore CTRL+C in the worker process.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':
    # Point cloud sampling count control parameters - modify here to control point cloud density
    # Note: larger values mean denser point clouds but longer processing time
    # Separate sampling densities for planar and curved surfaces
    # Curved surfaces (e.g., cylindrical holes) need denser sampling to capture geometry adequately
    FACE_POINTS_PER_SIDE_PLANAR = 8   # Planar face sampling: 6x6=36 points per planar face
    FACE_POINTS_PER_SIDE_CURVED = 8  # Curved face sampling: 10x10=100 points per curved face (more dense for circular holes)
    EDGE_POINTS_PER_EDGE = FACE_POINTS_PER_SIDE_PLANAR ** 2  # Edge point cloud sampling: based on planar face density
    
    # Parameters to be set before use
    dataset_scale = 'large' # or large or tiny
    num_features = 24 # for large dataset, use full classes
    
    # 新增:倒角和圆角的权重控制
    CHAMFER_ROUND_BOOST = False  # 是否提升倒角圆角概率
    CHAMFER_ROUND_WEIGHT = 1  # 倒角圆角的权重倍数(默认其他特征为1.0)
    
    # for tiny dataset, only common features
    # Through hole, Blind hole, Rectangular pocket, Rectangular through slot, Round, Chamfer
    tiny_dataset_cand_feats = [1, 12, 14, 6, 0, 23]
    cand_feat_weights = [0.6, 0.6, 1, 1, 2.0, 2.0]  # 提高倒角圆角权重(从0.3改为2.0)
    dataset_dir = 'data'
    combo_range = [5, 12]
    num_samples = 20000
    num_workers = 16

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    combos = []
    for idx in range(num_samples):
        num_inter_feat = random.randint(combo_range[0], combo_range[1])
        if dataset_scale == 'large':
            if CHAMFER_ROUND_BOOST:
                # 使用加权随机,提高倒角和圆角的生成概率
                all_features = list(range(num_features))  # [0,1,2,...,23]
                feature_weights = [1.0] * num_features    # 默认权重为1.0
                feature_weights[0] = CHAMFER_ROUND_WEIGHT   # chamfer权重提高
                feature_weights[23] = CHAMFER_ROUND_WEIGHT  # round权重提高
                combo = random.choices(all_features, weights=feature_weights, k=num_inter_feat)
            else:
                # 原来的均匀分布
                combo = [random.randint(0, num_features-1) for _ in range(num_inter_feat)] # no stock face
        elif dataset_scale == 'tiny':
            combo = random.choices(tiny_dataset_cand_feats, weights=cand_feat_weights, k=num_inter_feat)

        now =  time.localtime()
        now_time = time.strftime("%Y%m%d_%H%M%S", now)
        file_name = now_time + '_' + str(idx)
        combos.append((file_name, combo))

    if num_workers == 1:
        # Single worker with progress bar
        for combo in tqdm(combos, desc="Generating point clouds", unit="shape"):
            generate_shape((dataset_dir, combo))
    elif num_workers > 1: # multiprocessing
        pool = Pool(processes=num_workers, initializer=initializer)
        try:
            result = list(tqdm(pool.imap(generate_shape, [(dataset_dir, combo) for combo in combos]), 
                            desc="Generating point clouds", 
                            total=len(combos),
                            unit="shape"))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
    else:
        AssertionError('error number of workers')
    
    gc.collect()


