import vtk
import numpy as np
import logging
import math

try:
    import slicer
    from ScanToSocket import ScanToSocketLogic
except ImportError:
    pass # Allows some methods to be tested outside Slicer if needed

from SyntheticPatientGenerator import SyntheticPatientGenerator

class SyntheticEvaluator:
    """
    Evaluates the accuracy and robustness of the ScanToSocket algorithm using
    mathematically exact synthetic patient models.
    """
    
    def __init__(self):
        self.logic = ScanToSocketLogic()
        
    def _create_vtk_markups(self, p0, p1, p2=None):
        """Creates a vtkMRMLMarkupsFiducialNode for testing."""
        markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        markups.AddControlPoint(vtk.vtkVector3d(p0[0], p0[1], p0[2]))
        markups.AddControlPoint(vtk.vtkVector3d(p1[0], p1[1], p1[2]))
        if p2 is not None:
            markups.AddControlPoint(vtk.vtkVector3d(p2[0], p2[1], p2[2]))
        return markups

    def _create_model_node(self, polydata, name="SyntheticPatient"):
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        modelNode.SetAndObservePolyData(polydata)
        
        # Set some nice display properties
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetColor(0.8, 0.4, 0.4) # Fleshy red
        displayNode.SetOpacity(0.8)
        
        return modelNode

    def run_trajectory_evaluation(self, polydata, ground_truth, visualize=False):
        """
        Runs the marching algorithm and compares against ground truth trajectory.
        """
        modelNode = self._create_model_node(polydata, "SyntheticPatient_Eval")
        
        # Simulate perfect clinician clicks
        armLandmarks = self._create_vtk_markups(
            ground_truth["shoulder_pos"], 
            ground_truth["stump_tip"]
        )
        armLandmarks.SetName("Synthetic_ArmLandmarks")
        
        # Run algorithm and track time
        import time
        start_time = time.time()
        p1, dir_calc, r_calc, l_calc = self.logic.compute_stump_geometry(armLandmarks, modelNode, fallback_radius=40.0)
        end_time = time.time()
        comp_time_ms = (end_time - start_time) * 1000.0
        
        # Evaluate Trajectory Error
        dot_prod = abs(np.dot(dir_calc, ground_truth["trajectory_vector"]))
        angular_error = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
        
        # Evaluate Radius Error
        radius_error = abs(r_calc - ground_truth["effective_radius"])
        
        # Cleanup or Visualize
        if not visualize:
            slicer.mrmlScene.RemoveNode(modelNode)
            slicer.mrmlScene.RemoveNode(armLandmarks)
        else:
            # Generate the visual socket so the user can see what the module would build
            offset_distance = ground_truth["healthy_arm_length"]
            target_pos, p1, z_axis, y_axis, x_axis, scale, stump_radius, stump_length = self.logic.get_alignment_params(armLandmarks, offset_distance, modelNode)
            radius_elbow = scale * 27.5
            ext_dist = np.linalg.norm(np.array(target_pos) - np.array(p1))
            
            import math
            overlap_top = 15.0
            overlap_bottom = 15.0
            height_main = ext_dist + overlap_top
            total_height = height_main + overlap_bottom
            
            if height_main > 1e-6:
                slope = (stump_radius - radius_elbow) / height_main
                radius_bottom = radius_elbow - (slope * overlap_bottom)
            else:
                radius_bottom = radius_elbow
            radius_bottom = max(0.1, radius_bottom)
            
            conePolyData = self.logic.create_tapered_cylinder(stump_radius, radius_bottom, total_height, 50)
            
            direction = np.array(target_pos) - np.array(p1)
            direction = direction / np.linalg.norm(direction)
            default_axis = np.array([0.0, 1.0, 0.0])
            rotation_axis = np.cross(default_axis, direction)
            rot_norm = np.linalg.norm(rotation_axis)
            
            if rot_norm < 1e-6:
                rotation_axis = np.array([1.0, 0.0, 0.0])
                angle_deg = 0.0
            else:
                rotation_axis = rotation_axis / rot_norm
                angle_deg = math.degrees(math.acos(np.clip(np.dot(default_axis, direction), -1.0, 1.0)))
                
            top_point = p1 - (direction * overlap_top)
            bottom_point = target_pos + (direction * overlap_bottom)
            true_center = (top_point + bottom_point) / 2.0
            
            cylTransform = vtk.vtkTransform()
            cylTransform.Translate(true_center.tolist())
            cylTransform.RotateWXYZ(angle_deg, rotation_axis.tolist())
            
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(conePolyData)
            tf.SetTransform(cylTransform)
            tf.Update()
            
            cylinderModelNode = slicer.modules.models.logic().AddModel(tf.GetOutput())
            cylinderModelNode.SetName("Synthetic_GeneratedSocket")
            displayNode = cylinderModelNode.GetDisplayNode()
            if displayNode:
                displayNode.SetColor(0.2, 0.6, 1.0) # Blue
                displayNode.SetOpacity(0.5)

        return {
            "Angular_Error_deg": angular_error,
            "Radius_Error_mm": radius_error,
            "Calculated_Radius_mm": r_calc,
            "True_Effective_Radius_mm": ground_truth["effective_radius"],
            "Comp_Time_ms": comp_time_ms
        }

    def run_jitter_robustness_evaluation(self, polydata, ground_truth, noise_std_dev=2.0, visualize=False):
        """
        Adds noise to the input landmarks and measures drift in the final alignment output.
        """
        modelNode = self._create_model_node(polydata, "SyntheticPatient_Jitter")
        # In PopScanner, offset_distance is computed from the healthy arm length
        offset_distance = ground_truth["healthy_arm_length"]
        
        # Baseline (No Noise)
        cleanLandmarks = self._create_vtk_markups(ground_truth["shoulder_pos"], ground_truth["stump_tip"])
        target_clean, _, z_clean, _, _, _, _, _ = self.logic.get_alignment_params(cleanLandmarks, offset_distance, modelNode)
        
        # Noisy (Add noise to stump tip)
        noise = np.random.normal(0, noise_std_dev, 3)
        noisy_tip = ground_truth["stump_tip"] + noise
        
        noisyLandmarks = self._create_vtk_markups(ground_truth["shoulder_pos"], noisy_tip)
        noisyLandmarks.SetName("Synthetic_NoisyLandmarks")
        target_noisy, _, z_noisy, _, _, _, _, _ = self.logic.get_alignment_params(noisyLandmarks, offset_distance, modelNode)
        
        # Evaluate Drift
        translational_drift = np.linalg.norm(target_noisy - target_clean)
        dot_prod = abs(np.dot(z_clean, z_noisy))
        rotational_drift = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
        
        # Always clean up jitter landmarks to avoid overlapping points in visualization
        slicer.mrmlScene.RemoveNode(cleanLandmarks)
        slicer.mrmlScene.RemoveNode(noisyLandmarks)
        
        if not visualize:
            slicer.mrmlScene.RemoveNode(modelNode)
            
        return {
            "Input_Noise_Magnitude_mm": np.linalg.norm(noise),
            "Translational_Drift_mm": translational_drift,
            "Rotational_Drift_deg": rotational_drift
        }

    def visualize_case(self, case_name="High Eccentricity", ry=30, maj=50, min_rad=25, length=150.0):
        """
        Generates a single synthetic case and leaves the geometry and alignment in the Slicer scene
        so you can visually inspect the elliptical arm and the computed landmarks/axes.
        """
        print(f"Generating visual case: {case_name}")
        polydata, gt = SyntheticPatientGenerator.generate_elliptical_arm(
            torso_radius=150.0, arm_length=length, 
            major_radius=maj, minor_radius=min_rad, 
            rx=0.0, ry=ry, rz=0.0
        )
        
        # Call the evaluation but flag visualize=True so it doesn't delete the nodes
        res = self.run_trajectory_evaluation(polydata, gt, visualize=True)
        print(f"Visual Case - Angular Error: {res['Angular_Error_deg']:.2f} deg")
        print(f"Visual Case - Radius Error:  {res['Radius_Error_mm']:.2f} mm")
        
        # Center the 3D view on the new model
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()

    def run_suite(self, visualize=False):
        """
        Runs a suite of synthetic tests and prints a categorical table report.
        If visualize=True, leaves the models in the scene spaced apart for visual inspection.
        """
        import random
        random.seed(42) # Reproducible randomness
        
        print("==================================================================================================")
        print("                        POPSCANNER SYNTHETIC EVALUATION SUITE (30 Cases)                          ")
        print("==================================================================================================")
        
        # 10 of each category (Stratified Sampling)
        lengths = [random.uniform(70.0, 100.0) for _ in range(10)] + \
                  [random.uniform(100.0, 160.0) for _ in range(10)] + \
                  [random.uniform(160.0, 220.0) for _ in range(10)]
                  
        thicknesses = [random.uniform(30.0, 35.0) for _ in range(10)] + \
                      [random.uniform(35.0, 50.0) for _ in range(10)] + \
                      [random.uniform(50.0, 70.0) for _ in range(10)]
                      
        def rand_angle(min_sum, max_sum):
            s = random.uniform(min_sum, max_sum)
            rx = random.uniform(-s, s)
            ry = s - abs(rx)
            return rx, ry
            
        angles = [rand_angle(0, 29) for _ in range(10)] + \
                 [rand_angle(30, 70) for _ in range(10)] + \
                 [rand_angle(71, 120) for _ in range(10)]
                 
        random.shuffle(lengths)
        random.shuffle(thicknesses)
        random.shuffle(angles)
        
        test_cases = []
        for i in range(30):
            maj = thicknesses[i]
            minor = random.uniform(25.0, maj)
            test_cases.append({
                "name": f"Case {i+1}", 
                "rx": angles[i][0], "ry": angles[i][1], 
                "maj": maj, "min": minor, "len": lengths[i]
            })
        
        results = []
        offset_x = 0.0
        
        # We will suppress the individual case prints to avoid console spam,
        # but print a progress bar.
        import sys
        sys.stdout.write("Running 30 evaluation cases: [")
        sys.stdout.flush()
        
        for case in test_cases:
            polydata, gt = SyntheticPatientGenerator.generate_elliptical_arm(
                torso_radius=150.0, arm_length=case["len"], 
                major_radius=case["maj"], minor_radius=case["min"], 
                rx=case["rx"], ry=case["ry"], rz=0.0
            )
            
            if visualize:
                shift = vtk.vtkTransform()
                shift.Translate(offset_x, 0, 0)
                shifter = vtk.vtkTransformPolyDataFilter()
                shifter.SetInputConnection(polydata.GetOutputPort()) if hasattr(polydata, "GetOutputPort") else shifter.SetInputData(polydata)
                shifter.SetTransform(shift)
                shifter.Update()
                polydata = shifter.GetOutput()
                
                gt["shoulder_pos"] = gt["shoulder_pos"] + np.array([offset_x, 0, 0])
                gt["stump_tip"] = gt["stump_tip"] + np.array([offset_x, 0, 0])
            
            traj_res = self.run_trajectory_evaluation(polydata, gt, visualize=visualize)
            jitter_res = self.run_jitter_robustness_evaluation(polydata, gt, noise_std_dev=5.0, visualize=visualize)
            
            results.append({"case": case, "traj": traj_res, "jitter": jitter_res})
            offset_x += 650.0 # Increased spacing to prevent overlapping extended arms
            
            sys.stdout.write("=")
            sys.stdout.flush()
            
        print("] Done!\n")
        print("| CATEGORY                   | CNT | ANGULAR ERROR |  RADIUS ERROR   |  TRANS. DRIFT   |  ROT. DRIFT   |   COMP. TIME  |")
        print("|----------------------------|-----|---------------|-----------------|-----------------|---------------|---------------|")
        
        def print_bucket(name, condition):
            bucket = [r for r in results if condition(r["case"])]
            cnt = len(bucket)
            if cnt == 0: return
            ang = [r["traj"]["Angular_Error_deg"] for r in bucket]
            rad = [r["traj"]["Radius_Error_mm"] for r in bucket]
            tdr = [r["jitter"]["Translational_Drift_mm"] for r in bucket]
            rdr = [r["jitter"]["Rotational_Drift_deg"] for r in bucket]
            time_ms = [r["traj"]["Comp_Time_ms"] for r in bucket]
            
            ang_str = f"{np.mean(ang):.2f}° ± {np.std(ang):.2f}"
            rad_str = f"{np.mean(rad):.2f}mm ± {np.std(rad):.2f}"
            tdr_str = f"{np.mean(tdr):.2f}mm ± {np.std(tdr):.2f}"
            rdr_str = f"{np.mean(rdr):.2f}° ± {np.std(rdr):.2f}"
            time_str = f"{np.mean(time_ms):.1f}ms ± {np.std(time_ms):.1f}"
            
            print(f"| {name:<26} | {cnt:>3} | {ang_str:>13} | {rad_str:>15} | {tdr_str:>15} | {rdr_str:>13} | {time_str:>13} |")
            
        print_bucket("Length: Short (<100mm)", lambda c: c["len"] < 100)
        print_bucket("Length: Medium (100-160mm)", lambda c: 100 <= c["len"] <= 160)
        print_bucket("Length: Long (>160mm)", lambda c: c["len"] > 160)
        print("|----------------------------|-----|---------------|-----------------|-----------------|---------------|---------------|")
        print_bucket("Thickness: Thin (<35mm)", lambda c: c["maj"] < 35)
        print_bucket("Thickness: Normal (35-50mm)", lambda c: 35 <= c["maj"] <= 50)
        print_bucket("Thickness: Thick (>50mm)", lambda c: c["maj"] > 50)
        print("|----------------------------|-----|---------------|-----------------|-----------------|---------------|---------------|")
        print_bucket("Angle: Straight (<20°)", lambda c: abs(c["rx"]) + abs(c["ry"]) < 30)
        print_bucket("Angle: Moderate (20-60°)", lambda c: 30 <= abs(c["rx"]) + abs(c["ry"]) <= 70)
        print_bucket("Angle: Extreme (>60°)", lambda c: abs(c["rx"]) + abs(c["ry"]) > 70)
        print("|============================|=====|===============|=================|=================|===============|===============|")
        
        ang = [r["traj"]["Angular_Error_deg"] for r in results]
        rad = [r["traj"]["Radius_Error_mm"] for r in results]
        tdr = [r["jitter"]["Translational_Drift_mm"] for r in results]
        rdr = [r["jitter"]["Rotational_Drift_deg"] for r in results]
        time_ms = [r["traj"]["Comp_Time_ms"] for r in results]
        
        ang_str = f"{np.mean(ang):.2f}° ± {np.std(ang):.2f}"
        rad_str = f"{np.mean(rad):.2f}mm ± {np.std(rad):.2f}"
        tdr_str = f"{np.mean(tdr):.2f}mm ± {np.std(tdr):.2f}"
        rdr_str = f"{np.mean(rdr):.2f}° ± {np.std(rdr):.2f}"
        time_str = f"{np.mean(time_ms):.1f}ms ± {np.std(time_ms):.1f}"
        
        print(f"| OVERALL AVERAGES           |  30 | {ang_str:>13} | {rad_str:>15} | {tdr_str:>15} | {rdr_str:>13} | {time_str:>13} |")
        print("======================================================================================================================\n")

        if visualize:
            layoutManager = slicer.app.layoutManager()
            threeDWidget = layoutManager.threeDWidget(0)
            threeDView = threeDWidget.threeDView()
            threeDView.resetFocalPoint()

if __name__ == "__main__":
    evaluator = SyntheticEvaluator()
    evaluator.run_suite(visualize=False)
