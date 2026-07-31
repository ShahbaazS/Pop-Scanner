import vtk
import numpy as np

class SyntheticPatientGenerator:
    """
    Generates synthetic patient models (Torso + Elliptical Arm) with exact mathematical
    ground truth to rigorously evaluate the ScanToSocket trajectory and radius algorithms.
    """
    
    @staticmethod
    def generate_elliptical_arm(torso_radius=150.0, arm_length=150.0, major_radius=40.0, minor_radius=30.0, 
                                rx=0.0, ry=30.0, rz=0.0):
        """
        Generates a synthetic torso and elliptical arm.
        Returns:
            final_polydata: vtkPolyData of the combined mesh.
            ground_truth: Dictionary with exact analytical values.
        """
        # 1. Torso Sphere (Scaled to Ellipsoid Ribcage)
        torso_sphere = vtk.vtkSphereSource()
        torso_sphere.SetRadius(torso_radius)
        torso_sphere.SetThetaResolution(60)
        torso_sphere.SetPhiResolution(60)
        torso_sphere.Update()
        
        # Scale torso to make it look like a ribcage (flatter front-to-back, taller)
        torso_transform = vtk.vtkTransform()
        torso_transform.Scale(0.8, 0.5, 1.3)
        torso_transform_filter = vtk.vtkTransformPolyDataFilter()
        torso_transform_filter.SetInputConnection(torso_sphere.GetOutputPort())
        torso_transform_filter.SetTransform(torso_transform)
        torso_transform_filter.Update()
        
        torso_polydata = torso_transform_filter.GetOutputPort()
        
        # 2. Right Arm (Amputated / Measured Side)
        arm = vtk.vtkSphereSource()
        arm.SetRadius(1.0)
        arm.SetThetaResolution(60)
        arm.SetPhiResolution(60)
        arm.Update()
        
        # Transform logic (Applied bottom to top in code, right-to-left mathematically):
        transform = vtk.vtkTransform()
        # 5. Move to shoulder joint position on Torso surface (Placed far enough laterally to prevent straight-hanging arms from embedding into the ribcage)
        shoulder_offset = np.array([140.0, 0, torso_radius * 0.8])
        transform.Translate(shoulder_offset[0], shoulder_offset[1], shoulder_offset[2])
        # 4. Rotate arm at the shoulder joint
        transform.RotateZ(rz)
        transform.RotateX(rx)
        transform.RotateY(-ry) # Negative ry to point outwards (+X) away from Torso
        # 3. Shift the ellipsoid down so its Top Pole is exactly at the origin (0,0,0)
        transform.Translate(0, 0, -arm_length / 2.0)
        # 2. Scale to form a full ellipsoid
        # Total length from Top Pole to Bottom Pole will be exactly arm_length
        transform.Scale(major_radius, minor_radius, arm_length / 2.0)
        
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(arm.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        
        # 3. Left Arm (Healthy / Non-measured Side)
        left_arm = vtk.vtkSphereSource()
        left_arm.SetRadius(1.0)
        left_arm.SetThetaResolution(60)
        left_arm.SetPhiResolution(60)
        left_arm.Update()
        
        left_transform = vtk.vtkTransform()
        left_shoulder_offset = np.array([-140.0, 0, torso_radius * 0.8])
        left_transform.Translate(left_shoulder_offset[0], left_shoulder_offset[1], left_shoulder_offset[2])
        left_transform.RotateZ(-rz)
        left_transform.RotateX(rx)
        left_transform.RotateY(ry) # Positive ry to point outwards (-X)
        left_transform.Translate(0, 0, -(arm_length * 1.5) / 2.0)
        left_transform.Scale(major_radius, minor_radius, (arm_length * 1.5) / 2.0)
        
        left_transformFilter = vtk.vtkTransformPolyDataFilter()
        left_transformFilter.SetInputConnection(left_arm.GetOutputPort())
        left_transformFilter.SetTransform(left_transform)
        left_transformFilter.Update()
        
        # 3.5 Shoulder joint spheres to bridge arm and torso
        right_shoulder_sphere = vtk.vtkSphereSource()
        right_shoulder_sphere.SetRadius(45.0)
        right_shoulder_sphere.SetCenter(shoulder_offset)
        right_shoulder_sphere.SetThetaResolution(60)
        right_shoulder_sphere.SetPhiResolution(60)
        right_shoulder_sphere.Update()
        
        left_shoulder_sphere = vtk.vtkSphereSource()
        left_shoulder_sphere.SetRadius(45.0)
        left_shoulder_sphere.SetCenter(left_shoulder_offset)
        left_shoulder_sphere.SetThetaResolution(60)
        left_shoulder_sphere.SetPhiResolution(60)
        left_shoulder_sphere.Update()

        # 4. Combine into single mesh
        append = vtk.vtkAppendPolyData()
        append.AddInputConnection(torso_polydata)
        append.AddInputConnection(right_shoulder_sphere.GetOutputPort())
        append.AddInputConnection(transformFilter.GetOutputPort())
        append.AddInputConnection(left_shoulder_sphere.GetOutputPort())
        append.AddInputConnection(left_transformFilter.GetOutputPort())
        append.Update()
        
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(append.GetOutputPort())
        cleaner.Update()
        
        final_polydata = cleaner.GetOutput()
        
        # 4. Compute exact mathematical ground truth
        rot_transform = vtk.vtkTransform()
        rot_transform.RotateZ(rz)
        rot_transform.RotateX(rx)
        rot_transform.RotateY(-ry)
        
        # Local Z vector is (0,0,-1)
        dir_local = np.array([0.0, 0.0, -1.0, 0.0])
        dir_world = np.zeros(4)
        rot_transform.GetMatrix().MultiplyPoint(dir_local, dir_world)
        trajectory_vector = dir_world[0:3] / np.linalg.norm(dir_world[0:3])
        
        # Simulate clinical click exactly on the top surface of the shoulder sphere along the centerline axis
        shoulder_pos = shoulder_offset - trajectory_vector * 45.0
        # Tip is clicked at the distal pole of the stump
        stump_tip = shoulder_offset + trajectory_vector * arm_length
        
        # Effective radius approximation for an ellipse (geometric mean or perimeter approximation)
        # ScanToSocket uses median of bounding distances. 
        # Effective radius approximation for an ellipsoid tapering to a point.
        # The cross-sectional radius at distance Z from shoulder is R * sqrt(1 - (Z/L)^2).
        # The median radius along the length is roughly 86.6% of the base radius.
        base_avg_radius = (major_radius + minor_radius) / 2.0 
        effective_radius = base_avg_radius * 0.866
        
        ground_truth = {
            "shoulder_pos": shoulder_pos,
            "stump_tip": stump_tip,
            "trajectory_vector": trajectory_vector,
            "major_radius": float(major_radius),
            "minor_radius": float(minor_radius),
            "effective_radius": float(effective_radius),
            "healthy_arm_length": float(arm_length * 1.5),
            "arm_length": float(arm_length)
        }
        
        return final_polydata, ground_truth
