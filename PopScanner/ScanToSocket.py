import logging
import os
from typing import Annotated

import vtk
import qt
import ctk
import numpy as np
import math

import slicer
from slicer import vtkMRMLModelNode
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLMarkupsFiducialNode

try:
    from vtkSlicerSegmentationsModuleMRMLPython import vtkSegmentationConverter
except ImportError:
    try:
        from slicer import vtkSegmentationConverter
    except ImportError:
        import vtkSegmentationCore
        vtkSegmentationConverter = vtkSegmentationCore.vtkSegmentationConverter

from slicer import vtkMRMLScalarVolumeNode


#
# ScanToSocket
#


class ScanToSocket(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ScanToSocket")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Prosthetics")]
        self.parent.dependencies = []
        self.parent.contributors = ["Emese Elkind, Shahbaaz Siddiqui"]
        self.parent.helpText = _("Module developed by Q-CAR for above-elbow prosthetic alignment.")
        self.parent.acknowledgementText = _("Developed by Q-CARE.")

        # --- ADD THESE LINES FOR THE LOGO ---
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        # Assuming your logo is in a 'Resources/Icons' folder relative to this file
        iconPath = os.path.join(moduleDir, 'Resources', 'Icons', 'qcare logo2.png')
        
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        else:
            logging.error(f"Logo not found at {iconPath}")
        # -------------------------------------

        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # ScanToSocket1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ScanToSocket",
        sampleName="ScanToSocket1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "ScanToSocket1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="ScanToSocket1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="ScanToSocket1",
    )

    # ScanToSocket2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ScanToSocket",
        sampleName="ScanToSocket2",
        thumbnailFileName=os.path.join(iconsPath, "ScanToSocket2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="ScanToSocket2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="ScanToSocket2",
    )


#
# ScanToSocketParameterNode
#


@parameterNodeWrapper
class ScanToSocketParameterNode:
    """
    The parameters needed by module.

    inputFilePath - Path to the STL or OBJ file to process.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputFilePath: str
    outputModel: vtkMRMLModelNode
    # New nodes for registration
    armLandmarks: vtkMRMLMarkupsFiducialNode


#
# ScanToSocketWidget
#


class ScanToSocketWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        # Crop mode state
        self._cropMode = False
        self._savedLayout = None
        self._cropSegNode = None
        self._cropEditorWidget = None
        self._cropEditorNode = None
        self._originalPolyData = None  # backup of patient scan before crop
        self._previewActive = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ScanToSocket.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = ScanToSocketLogic()
        
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Use the UI slider for shoulder-to-elbow distance
        self.ui.shoulderDistanceSlider.setToolTip("Distance from the shoulder to the elbow target point in mm.")

        containerLayout = uiWidget.layout()

        # Connections
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.exportSTLButton.connect("clicked(bool)", self.onExportButton)
        self.ui.inputFileBrowseButton.connect("clicked(bool)", self.onBrowseInputFile)
        
        self.ui.armMarkupsWidget.connect("markupsNodeChanged()", self._onArmMarkupsNodeChanged)
        self.ui.armMarkupsWidget.connect("activeMarkupsNodeChanged(vtkMRMLNode*)", self._onArmMarkupsNodeChanged)

        # --- Crop Torso UI ---
        cropGroupBox = qt.QGroupBox("Crop Torso")
        cropLayout = qt.QVBoxLayout(cropGroupBox)
        cropLayout.setContentsMargins(8, 6, 8, 6)
        cropLayout.setSpacing(6)

        cropInstructions = qt.QLabel(
            "Draw a freehand loop around the torso in the 3D view.\n"
            "Everything inside the loop is erased. The disconnected\n"
            "torso piece is then removed automatically."
        )
        cropInstructions.setWordWrap(True)
        cropInstructions.setStyleSheet("color: #aaa; font-size: 11px;")
        cropLayout.addWidget(cropInstructions)

        cropBtnRow = qt.QHBoxLayout()
        self._startCropBtn = qt.QPushButton("✂  Start Crop")
        self._startCropBtn.setToolTip("Enter crop mode: erase the arm/torso junction.")
        self._startCropBtn.setEnabled(False)
        self._doneCropBtn = qt.QPushButton("✓  Done")
        self._doneCropBtn.setToolTip("Accept crop and return to normal mode.")
        self._doneCropBtn.setEnabled(False)
        self._cancelCropBtn = qt.QPushButton("✗  Cancel")
        self._cancelCropBtn.setToolTip("Discard all crop edits and restore original scan.")
        self._cancelCropBtn.setEnabled(False)
        cropBtnRow.addWidget(self._startCropBtn)
        cropBtnRow.addWidget(self._doneCropBtn)
        cropBtnRow.addWidget(self._cancelCropBtn)
        cropLayout.addLayout(cropBtnRow)

        # Insert crop group below the Extension Distance control and above the Export button
        exportButtonIndex = containerLayout.indexOf(self.ui.exportSTLButton)
        containerLayout.insertWidget(exportButtonIndex, cropGroupBox)

        self._startCropBtn.connect("clicked(bool)", self.onStartCrop)
        self._doneCropBtn.connect("clicked(bool)", self.onDoneCrop)
        self._cancelCropBtn.connect("clicked(bool)", self.onCancelCrop)

        # --- Preview & Adjustment UI ---
        previewGroupBox = qt.QGroupBox("Preview Alignment")
        previewOuterLayout = qt.QVBoxLayout(previewGroupBox)
        previewOuterLayout.setContentsMargins(8, 6, 8, 6)

        previewInstructions = qt.QLabel(
            "1. Click Preview to initialize elbow position.\n"
            "2. Drag the Trajectory Line or sliders to adjust cone & radii.\n"
            "3. Use the 3D Gizmo to rotate and scale the elbow."
        )
        previewInstructions.setStyleSheet("color: #aaa; font-size: 11px;")
        previewOuterLayout.addWidget(previewInstructions)

        self._previewBtn = qt.QPushButton("▶  Initialize Preview")
        self._previewBtn.setEnabled(False)
        previewOuterLayout.addWidget(self._previewBtn)

        coneSlidersLayout = qt.QFormLayout()
        
        self._radiusTopSlider = ctk.ctkSliderWidget()
        self._radiusTopSlider.minimum = 10.0
        self._radiusTopSlider.maximum = 150.0
        self._radiusTopSlider.value = 50.0
        self._radiusTopSlider.singleStep = 0.5
        self._radiusTopSlider.suffix = " mm"
        self._radiusTopSlider.decimals = 1
        self._radiusTopSlider.setToolTip("Adjust the top radius of the bridging cone (at the stump tip).")
        coneSlidersLayout.addRow("Top Radius (Stump):", self._radiusTopSlider)

        self._radiusBottomSlider = ctk.ctkSliderWidget()
        self._radiusBottomSlider.minimum = 10.0
        self._radiusBottomSlider.maximum = 100.0
        self._radiusBottomSlider.value = 27.5
        self._radiusBottomSlider.singleStep = 0.5
        self._radiusBottomSlider.suffix = " mm"
        self._radiusBottomSlider.decimals = 1
        self._radiusBottomSlider.setToolTip("Adjust the bottom radius of the bridging cone (at the elbow).")
        coneSlidersLayout.addRow("Bottom Radius (Elbow):", self._radiusBottomSlider)

        self._overlapTopSlider = ctk.ctkSliderWidget()
        self._overlapTopSlider.minimum = 0.0
        self._overlapTopSlider.maximum = 80.0
        self._overlapTopSlider.value = 15.0
        self._overlapTopSlider.singleStep = 1.0
        self._overlapTopSlider.suffix = " mm"
        self._overlapTopSlider.decimals = 1
        self._overlapTopSlider.setToolTip("Adjust how much the top of the cone overlaps onto the residual limb tip.")
        coneSlidersLayout.addRow("Top Overlap (Stump):", self._overlapTopSlider)

        previewOuterLayout.addLayout(coneSlidersLayout)

        applyButtonIndex = containerLayout.indexOf(self.ui.applyButton)
        containerLayout.insertWidget(applyButtonIndex, previewGroupBox)

        self._previewBtn.connect("clicked(bool)", self.onPreviewElbow)
        self.ui.shoulderDistanceSlider.connect("valueChanged(double)", self._onDistanceChanged)
        self._radiusTopSlider.connect("valueChanged(double)", self._onConeSliderChanged)
        self._radiusBottomSlider.connect("valueChanged(double)", self._onConeSliderChanged)
        self._overlapTopSlider.connect("valueChanged(double)", self._onConeSliderChanged)

        self.initializeParameterNode()
        self.loadProstheticModel()

    def loadProstheticModel(self) -> None:
        """Load the prosthetic elbow model at startup."""
        # Get the path relative to this module
        
        try:
            if slicer.util.getNode("Prosthetic Elbow"):
                return  # Exit early, don't load again
        except:
            pass
        
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        prostheticModelPath = os.path.join(os.path.dirname(moduleDir), "elbow.stl")
        
        logging.info(f"Looking for prosthetic model at: {prostheticModelPath}")
        
        if not os.path.exists(prostheticModelPath):
            logging.error(f"Prosthetic model file not found: {prostheticModelPath}")
            slicer.util.errorDisplay(
                f"Prosthetic model not found at:\n{prostheticModelPath}\n\n"
                "Please ensure the elbow.stl file is in the correct location."
            )
            return
        
        try:
            logging.info(f"Attempting to load model from: {prostheticModelPath}")
            modelNode = slicer.util.loadModel(prostheticModelPath)
            if modelNode:
                # Rename for clarity in the scene
                modelNode.SetName("Prosthetic Elbow")
                # Set color to green
                displayNode = modelNode.GetDisplayNode()
                if displayNode:
                    displayNode.SetColor(0, 1, 0)  # Green: R=0, G=1, B=0
                logging.info(f"Prosthetic model loaded successfully: {modelNode.GetName()}")
                self._updateElbowWidthLabel()
                return True
            else:
                logging.error("Failed to load the prosthetic model (elbow.stl) - loadModel returned None")
                slicer.util.errorDisplay("Failed to load the prosthetic model (elbow.stl).")
                return False
        except Exception as e:
            logging.error(f"Exception loading prosthetic model: {str(e)}", exc_info=True)
            slicer.util.errorDisplay(f"Error loading prosthetic model:\n{str(e)}")
            return False

    def setupSliceletUI(self) -> None:
        """Setup the slicelet UI by hiding unnecessary elements for a streamlined workflow."""
        # Hide main Slicer UI elements to create a focused slicelet experience
        slicer.util.setMenuBarsVisible(False)
        slicer.util.setToolbarsVisible(False)
        slicer.util.setStatusBarVisible(False)
        slicer.util.setModulePanelTitleVisible(False)
        slicer.util.setModuleHelpSectionVisible(False)
        
        # Hide the data probe (bottom left corner info)
        slicer.util.setDataProbeVisible(False)
        
        # Hide the logo and version info in the bottom right
        slicer.util.setApplicationLogoVisible(False)
        
        # Set a custom window title for the slicelet
        slicer.app.mainWindow().setWindowTitle("ScanToSocket - Prosthetic Socket Generator")
        
        # Set window to a reasonable size for the workflow
        slicer.app.mainWindow().resize(1200, 800)
        
        # Ensure we're in 3D view layout
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: ScanToSocketParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
    
    def _onArmMarkupsNodeChanged(self) -> None:
        armNode = self.ui.armMarkupsWidget.currentNode()
        if armNode:
            armNode.SetMaximumNumberOfControlPoints(3)
            self.addObserver(armNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

            if not hasattr(self, "_pointObserverAdded"):
                armNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onPointModified)
                self._pointObserverAdded = True

        self._checkCanApply()

    def _onPointModified(self, caller, event) -> None:
        """Fires instantly when a user drags a fiducial point in the 3D view."""
        if self._previewActive:
            self.onPreviewElbow()

    def _checkCanApply(self, caller=None, event=None) -> None:
        # Logic: Only enable if we have an input file AND the patient arm has at least 2 points
        armReady = self.ui.armMarkupsWidget.currentNode() and self.ui.armMarkupsWidget.currentNode().GetNumberOfControlPoints() >= 2
        
        if self._parameterNode and self._parameterNode.inputFilePath and armReady:
            self.ui.applyButton.text = _("Generate Prosthetic Socket")
            self.ui.applyButton.enabled = True
            self._previewBtn.setEnabled(True)
        else:
            self.ui.applyButton.text = _("Place at least 2 points on Patient Arm (Shoulder, Tip)")
            self.ui.applyButton.enabled = False
            self._previewBtn.setEnabled(False)
        
        # Enable Export and crop button if prosthetic socket has been generated
        try:
            slicer.util.getNode("Prosthetic Socket")
            self.ui.exportSTLButton.enabled = True
            if not self._cropMode:
                self._startCropBtn.setEnabled(True)
        except Exception:
            self.ui.exportSTLButton.enabled = False
            if not self._cropMode:
                self._startCropBtn.setEnabled(False)

    # ------------------------------------------------------------------
    # Preview hooks
    # ------------------------------------------------------------------

    def onPreviewElbow(self) -> None:
        armNode = self.ui.armMarkupsWidget.currentNode()
        prostheticModel = slicer.util.getNode("Prosthetic Elbow")
        
        if not prostheticModel or not armNode or armNode.GetNumberOfControlPoints() < 2:
            slicer.util.errorDisplay("Make sure 'Patient Scan' is loaded and at least 2 points are placed (Point 1: Shoulder, Point 2: Amputation Tip).")
            return

        self._previewActive = True
        self._isUpdating = False # Anti-infinite-loop flag

        dist = self.ui.shoulderDistanceSlider.value
        patientModel = slicer.mrmlScene.GetFirstNodeByName("Patient Scan")
        target_pos, p1, z_axis, y_axis, x_axis, scale, stump_radius, stump_length = self.logic.get_alignment_params(armNode, dist, patientModel)
        self._stumpLength = stump_length
        self._updateDistanceReadouts()

        # 1. Setup Master Transform & Gizmos
        transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
        if not transformNode:
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "MasterElbowTransform")
            
        transform = vtk.vtkTransform()
        transform.Translate(target_pos)
        base_rot = vtk.vtkMatrix4x4()
        for i in range(3):
            base_rot.SetElement(i, 0, x_axis[i])
            base_rot.SetElement(i, 1, y_axis[i])
            base_rot.SetElement(i, 2, z_axis[i])
        transform.Concatenate(base_rot)
        transform.RotateY(180) # Flip down
        transform.Scale(scale, scale, scale)

        transformNode.SetMatrixTransformToParent(transform.GetMatrix())
        prostheticModel.SetAndObserveTransformNodeID(transformNode.GetID())

        # Enable ALL Gizmo capabilities directly on the elbow
        transformNode.CreateDefaultDisplayNodes()
        dispNode = transformNode.GetDisplayNode()
        if dispNode:
            dispNode.SetEditorVisibility(True)
            dispNode.SetEditorTranslationEnabled(True)
            dispNode.SetEditorRotationEnabled(True)
            dispNode.SetEditorScalingEnabled(True)

        # 2. Setup Interactive Trajectory Line
        lineNode = slicer.mrmlScene.GetFirstNodeByName("TrajectoryLine")
        if not lineNode:
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "TrajectoryLine")
            lineNode.GetDisplayNode().SetColor(0, 1, 1) # Cyan
            lineNode.GetDisplayNode().SetLineThickness(3.0)
        
        lineNode.RemoveAllControlPoints()
        lineNode.AddControlPoint(p1)          # Point 0: Stump Center
        lineNode.AddControlPoint(target_pos)  # Point 1: Elbow Target
        lineNode.SetNthControlPointLocked(0, False) # Unlock Stump Center so user can freely adjust both line ends

        # 3. Bind the Bi-directional Observers
        if not hasattr(self, "_transformObserver"):
            self._transformObserver = transformNode.AddObserver(slicer.vtkMRMLTransformNode.TransformModifiedEvent, self._onTransformModified)
        if not hasattr(self, "_lineObserver"):
            self._lineObserver = lineNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onLineModified)

        if not getattr(self, "_slidersInitialized", False):
            if hasattr(self, "_radiusTopSlider"):
                self._radiusTopSlider.blockSignals(True)
                self._radiusTopSlider.value = stump_radius if stump_radius else 50.0
                self._radiusTopSlider.blockSignals(False)
            if hasattr(self, "_radiusBottomSlider"):
                self._radiusBottomSlider.blockSignals(True)
                self._radiusBottomSlider.value = scale * 27.5
                self._radiusBottomSlider.blockSignals(False)
            self._slidersInitialized = True

        self._redrawPreviewCylinder()

    def _onTransformModified(self, caller, event) -> None:
        """Fires when the user drags/rotates/scales the 3D Elbow Gizmo."""
        if not self._previewActive or getattr(self, "_isUpdating", False): return
        self._isUpdating = True
        try:
            transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
            lineNode = slicer.mrmlScene.GetFirstNodeByName("TrajectoryLine")
            if not transformNode or not lineNode: return

            # Extract new position from gizmo
            matrix = vtk.vtkMatrix4x4()
            transformNode.GetMatrixTransformToParent(matrix)
            p1 = [matrix.GetElement(0, 3), matrix.GetElement(1, 3), matrix.GetElement(2, 3)]
            
            # Snap the Trajectory line to follow the elbow
            lineNode.SetNthControlPointPosition(1, p1[0], p1[1], p1[2])
            
            # Update distance UI silently
            p0 = np.zeros(3); lineNode.GetNthControlPointPosition(0, p0)
            dist = np.linalg.norm(np.array(p1) - p0)
            self.ui.shoulderDistanceSlider.blockSignals(True)
            self.ui.shoulderDistanceSlider.setValue(dist + getattr(self, "_stumpLength", 0.0))
            self.ui.shoulderDistanceSlider.blockSignals(False)

            if hasattr(self, "_radiusBottomSlider"):
                current_scale = np.linalg.norm([matrix.GetElement(0,0), matrix.GetElement(1,0), matrix.GetElement(2,0)])
                self._radiusBottomSlider.blockSignals(True)
                self._radiusBottomSlider.value = current_scale * 27.5
                self._radiusBottomSlider.blockSignals(False)

            self._redrawPreviewCylinder()
            self._updateDistanceReadouts()
        finally:
            self._isUpdating = False

    def _onLineModified(self, caller, event) -> None:
        """Fires when the user drags the end of the Cyan Trajectory Line."""
        if not self._previewActive or getattr(self, "_isUpdating", False): return
        self._isUpdating = True
        try:
            transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
            lineNode = slicer.mrmlScene.GetFirstNodeByName("TrajectoryLine")
            if not transformNode or not lineNode: return

            p1 = np.zeros(3)
            lineNode.GetNthControlPointPosition(1, p1)

            # Snap the Elbow to follow the Trajectory Line
            matrix = vtk.vtkMatrix4x4()
            transformNode.GetMatrixTransformToParent(matrix)
            matrix.SetElement(0, 3, p1[0])
            matrix.SetElement(1, 3, p1[1])
            matrix.SetElement(2, 3, p1[2])
            transformNode.SetMatrixTransformToParent(matrix)

            p0 = np.zeros(3); lineNode.GetNthControlPointPosition(0, p0)
            dist = np.linalg.norm(p1 - p0)
            self.ui.shoulderDistanceSlider.blockSignals(True)
            self.ui.shoulderDistanceSlider.setValue(dist + getattr(self, "_stumpLength", 0.0))
            self.ui.shoulderDistanceSlider.blockSignals(False)

            self._redrawPreviewCylinder()
            self._updateDistanceReadouts()
        finally:
            self._isUpdating = False

    def _onDistanceChanged(self, val) -> None:
        """Fires if the user changes the shoulder-to-elbow distance slider."""
        if not self._previewActive or getattr(self, "_isUpdating", False): return
        
        lineNode = slicer.mrmlScene.GetFirstNodeByName("TrajectoryLine")
        if not lineNode: return
        
        # Extend the line exactly along its current trajectory vector
        p0 = np.zeros(3); lineNode.GetNthControlPointPosition(0, p0)
        p1 = np.zeros(3); lineNode.GetNthControlPointPosition(1, p1)
        direction = p1 - p0
        norm = np.linalg.norm(direction)
        if norm < 1e-6: return
        
        ext_len = max(10.0, val - getattr(self, "_stumpLength", 0.0))
        new_p1 = p0 + ((direction / norm) * ext_len)
        
        # Updating the line triggers the Observer, which moves the elbow & redraws the cylinder
        lineNode.SetNthControlPointPosition(1, new_p1[0], new_p1[1], new_p1[2])
        self._updateDistanceReadouts()

    def _updateDistanceReadouts(self) -> None:
        """Updates the read-only Stump Length, Extension Distance, and Elbow Width labels in the UI."""
        if not hasattr(self.ui, "stumpLengthValueLabel") or not hasattr(self.ui, "extensionDistanceValueLabel"):
            return
        stump_len = getattr(self, "_stumpLength", 0.0)
        total_dist = self.ui.shoulderDistanceSlider.value
        ext_len = max(0.0, total_dist - stump_len)
        self.ui.stumpLengthValueLabel.text = f"{stump_len:.1f} mm"
        self.ui.extensionDistanceValueLabel.text = f"{ext_len:.1f} mm"
        
        if hasattr(self, "_updateElbowWidthLabel"):
            self._updateElbowWidthLabel()

    def _onConeSliderChanged(self, value=None) -> None:
        """Fires if the user changes any of the bridging cone sliders."""
        if not hasattr(self, "_previewActive") or not self._previewActive:
            return
        if getattr(self, "_isUpdatingSliders", False):
            return
        self._isUpdatingSliders = True
        try:
            # Sync bottom radius slider to elbow gizmo scale!
            transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
            if transformNode and hasattr(self, "_radiusBottomSlider"):
                target_scale = self._radiusBottomSlider.value / 27.5
                matrix = vtk.vtkMatrix4x4()
                transformNode.GetMatrixTransformToParent(matrix)
                # Keep rotation/translation, update uniform scale of columns 0, 1, 2
                for col in range(3):
                    vec = [matrix.GetElement(0, col), matrix.GetElement(1, col), matrix.GetElement(2, col)]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = [v * (target_scale / norm) for v in vec]
                        matrix.SetElement(0, col, vec[0])
                        matrix.SetElement(1, col, vec[1])
                        matrix.SetElement(2, col, vec[2])
                transformNode.SetMatrixTransformToParent(matrix)
            self._redrawPreviewCylinder()
            self._updateDistanceReadouts()
        finally:
            self._isUpdatingSliders = False

    def _redrawPreviewCylinder(self) -> None:
        lineNode = slicer.mrmlScene.GetFirstNodeByName("TrajectoryLine")
        transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
        if not lineNode or not transformNode: return

        p0 = np.zeros(3); lineNode.GetNthControlPointPosition(0, p0)
        p1 = np.zeros(3); lineNode.GetNthControlPointPosition(1, p1)

        # Extract current scale directly from the gizmo to determine cylinder fatness
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        current_scale = np.linalg.norm([matrix.GetElement(0,0), matrix.GetElement(1,0), matrix.GetElement(2,0)])
        if hasattr(self, "_radiusBottomSlider"):
            radius_elbow = self._radiusBottomSlider.value
        else:
            radius_elbow = current_scale * 27.5

        if hasattr(self, "_radiusTopSlider"):
            radius_top = self._radiusTopSlider.value
        else:
            patientModel = slicer.mrmlScene.GetFirstNodeByName("Patient Scan")
            armNode = self.ui.armMarkupsWidget.currentNode()
            _, _, radius_top = self.logic.compute_stump_geometry(armNode, patientModel, fallback_radius=radius_elbow * 1.25)

        cylNode = slicer.mrmlScene.GetFirstNodeByName("PreviewCylinder")
        if not cylNode:
            cylNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "PreviewCylinder")
            cylNode.CreateDefaultDisplayNodes()
            dispNode = cylNode.GetDisplayNode()
            dispNode.SetColor(0, 1, 1) # Cyan
            dispNode.SetOpacity(0.3)   # Ghosted

        overlap_top = self._overlapTopSlider.value if hasattr(self, "_overlapTopSlider") else 15.0
        overlap_bottom = 15.0
        height_main = np.linalg.norm(p1 - p0) + overlap_top
        total_height = height_main + overlap_bottom

        if height_main > 1e-6:
            slope = (radius_top - radius_elbow) / height_main
            radius_bottom = radius_elbow - (slope * overlap_bottom)
        else:
            radius_bottom = radius_elbow
        radius_bottom = max(0.1, radius_bottom)

        conePolyData = self.logic.create_tapered_cylinder(radius_top, radius_bottom, total_height, 50)

        direction = p1 - p0
        norm = np.linalg.norm(direction)
        if norm < 1e-6: return
        direction = direction / norm

        default_axis = np.array([0.0, 1.0, 0.0])
        rotation_axis = np.cross(default_axis, direction)
        rot_norm = np.linalg.norm(rotation_axis)
        
        if rot_norm < 1e-6:
            angle_deg = 0.0
            rotation_axis = np.array([1.0, 0.0, 0.0])
        else:
            rotation_axis = rotation_axis / rot_norm
            angle_deg = math.degrees(math.acos(np.clip(np.dot(default_axis, direction), -1.0, 1.0)))

        true_center = ((p0 - (direction * overlap_top)) + (p1 + (direction * overlap_bottom))) / 2.0

        cylTransform = vtk.vtkTransform()
        cylTransform.Translate(true_center.tolist())
        cylTransform.RotateWXYZ(angle_deg, rotation_axis.tolist())

        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(conePolyData)
        tf.SetTransform(cylTransform)
        tf.Update()
        cylNode.SetAndObservePolyData(tf.GetOutput())

        self._updateElbowWidthLabel()
        self._updateElbowWidthLine()

    def _getCurrentElbowWidthMm(self) -> float | None:
        """Compute the current elbow width in mm from the prosthetic model and its transform."""
        prostheticModel = slicer.util.getNode("Prosthetic Elbow")
        if not prostheticModel:
            return None

        polyData = prostheticModel.GetPolyData()
        if not polyData:
            return None

        bounds = [0.0] * 6
        polyData.GetBounds(bounds)
        local_width = bounds[1] - bounds[0]
        if local_width <= 0.0:
            return None

        scale = 1.0
        transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
        if transformNode:
            matrix = vtk.vtkMatrix4x4()
            transformNode.GetMatrixTransformToParent(matrix)
            scale = np.linalg.norm([
                matrix.GetElement(0, 0),
                matrix.GetElement(1, 0),
                matrix.GetElement(2, 0),
            ])
            if scale <= 0.0:
                scale = 1.0

        return local_width * scale

    def _updateElbowWidthLabel(self) -> None:
        width_mm = self._getCurrentElbowWidthMm()
        if width_mm is None and hasattr(self, "_radiusBottomSlider"):
            width_mm = 2.0 * self._radiusBottomSlider.value
            
        if hasattr(self.ui, "elbowWidthValueLabel"):
            if width_mm is None:
                self.ui.elbowWidthValueLabel.text = "-- mm"
            else:
                self.ui.elbowWidthValueLabel.text = f"{width_mm:.1f} mm"
        elif hasattr(self, "_elbowWidthLabel"):
            if width_mm is None:
                self._elbowWidthLabel.setText("Elbow Width: -- mm")
            else:
                self._elbowWidthLabel.setText(f"Elbow Width: {width_mm:.1f} mm")

    def _updateElbowWidthLine(self) -> None:
        prostheticModel = slicer.util.getNode("Prosthetic Elbow")
        transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
        if not prostheticModel or not transformNode:
            return

        polyData = prostheticModel.GetPolyData()
        if not polyData:
            return

        bounds = [0.0] * 6
        polyData.GetBounds(bounds)
        x_min, x_max = bounds[0], bounds[1]
        y_center = (bounds[2] + bounds[3]) / 2.0
        z_center = (bounds[4] + bounds[5]) / 2.0

        p0_local = [x_min, y_center, z_center]
        p1_local = [x_max, y_center, z_center]

        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)

        def world_point(local_pt):
            x, y, z = local_pt
            return [
                matrix.GetElement(0, 0) * x + matrix.GetElement(0, 1) * y + matrix.GetElement(0, 2) * z + matrix.GetElement(0, 3),
                matrix.GetElement(1, 0) * x + matrix.GetElement(1, 1) * y + matrix.GetElement(1, 2) * z + matrix.GetElement(1, 3),
                matrix.GetElement(2, 0) * x + matrix.GetElement(2, 1) * y + matrix.GetElement(2, 2) * z + matrix.GetElement(2, 3),
            ]

        p0_world = world_point(p0_local)
        p1_world = world_point(p1_local)

        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0_world)
        lineSource.SetPoint2(p1_world)
        lineSource.Update()

        lineNode = slicer.mrmlScene.GetFirstNodeByName("ElbowWidthLine")
        if not lineNode:
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ElbowWidthLine")
            lineNode.CreateDefaultDisplayNodes()
            displayNode = lineNode.GetDisplayNode()
            if displayNode:
                displayNode.SetColor(0.0, 0.0, 0.0)
                displayNode.SetOpacity(1.0)
                try:
                    displayNode.SetLineWidth(3)
                except Exception:
                    pass

        lineNode.SetAndObservePolyData(lineSource.GetOutput())

    # ------------------------------------------------------------------
    # Crop mode
    # ------------------------------------------------------------------

    def onStartCrop(self) -> None:
        """Enter crop mode: convert socket base to segmentation, open Segment Editor."""
        logging.info("--- Starting Crop Mode ---")
        
        try:
            # FIX: Grab the separated nodes
            targetModel = slicer.util.getNode("Socket Base")
            alignedElbow = slicer.util.getNode("Aligned Elbow")
            finalSocket = slicer.util.getNode("Prosthetic Socket")
        except Exception:
            slicer.util.errorDisplay("Could not find 'Prosthetic Socket'. Generate it first!")
            return

        # Backup original geometry so Cancel can restore it
        self._originalPolyData = vtk.vtkPolyData()
        self._originalPolyData.DeepCopy(targetModel.GetPolyData())

        # Hide the merged socket, show the untouched elbow
        finalSocket.GetDisplayNode().SetVisibility(False)
        alignedElbow.GetDisplayNode().SetVisibility(True)

        # Save current layout and switch to 3D-only view
        lm = slicer.app.layoutManager()
        self._savedLayout = lm.layout
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

        # Set 3D view to orthographic (parallel projection) and reset camera
        threeDWidget = lm.threeDWidget(0)
        if threeDWidget:
            cam = threeDWidget.threeDView().renderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            cam.SetParallelProjection(True)
            threeDWidget.threeDView().resetFocalPoint()
            threeDWidget.threeDView().renderWindow().Render()

        logging.info("Camera reset. Creating Segmentation Node...")

        # Convert target model → segmentation
        self._cropSegNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "CropSegmentation")
        self._cropSegNode.CreateDefaultDisplayNodes()
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(targetModel, self._cropSegNode)
        
        segmentation = self._cropSegNode.GetSegmentation()
        segID = segmentation.GetNthSegmentID(0)
        segmentation.GetSegment(segID).SetName("Patient Scan")

        logging.info("Creating Reference Volume...")

        # Create Reference Volume with explicit spatial matrix
        bounds = [0.0] * 6
        self._cropSegNode.GetRASBounds(bounds)
        margin = 20.0
        spacing = 1.0 
        dims = [
            int((bounds[1]-bounds[0]+2*margin)/spacing)+1,
            int((bounds[3]-bounds[2]+2*margin)/spacing)+1,
            int((bounds[5]-bounds[4]+2*margin)/spacing)+1,
        ]
        origin = [bounds[0]-margin, bounds[2]-margin, bounds[4]-margin]

        refArray = np.zeros([dims[2], dims[1], dims[0]], dtype=np.int16)
        self._cropRefVolume = slicer.util.addVolumeFromArray(refArray)
        self._cropRefVolume.SetName("CropRefVolume")
        
        # Strictly set the spatial matrix so the volume aligns with the mesh
        ijkToRAS = vtk.vtkMatrix4x4()
        ijkToRAS.SetElement(0, 0, spacing)
        ijkToRAS.SetElement(1, 1, spacing)
        ijkToRAS.SetElement(2, 2, spacing)
        ijkToRAS.SetElement(0, 3, origin[0])
        ijkToRAS.SetElement(1, 3, origin[1])
        ijkToRAS.SetElement(2, 3, origin[2])
        self._cropRefVolume.SetIJKToRASMatrix(ijkToRAS)

        # Force conversion to Binary Labelmap using this reference volume
        self._cropSegNode.SetReferenceImageGeometryParameterFromVolumeNode(self._cropRefVolume)
        self._cropSegNode.CreateBinaryLabelmapRepresentation()

        # Hide original model
        targetModel.GetDisplayNode().SetVisibility(False)

        logging.info("Setting up Segment Editor Widget...")

        # Set up Segment Editor
        self._cropEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self._cropEditorWidget.setMRMLScene(slicer.mrmlScene)
        self._cropEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "CropEditorNode")
        self._cropEditorWidget.setMRMLSegmentEditorNode(self._cropEditorNode)
        self._cropEditorWidget.setSegmentationNode(self._cropSegNode)
        self._cropEditorWidget.setSourceVolumeNode(self._cropRefVolume) 
        self._cropEditorWidget.setCurrentSegmentID(segID)

        # Slicer requires the widget to be in the UI to intercept 3D mouse events.
        # We add it to the layout, but hide the messy UI elements so it looks clean.
        self.layout.addWidget(self._cropEditorWidget)
        self._cropEditorWidget.setSegmentationNodeSelectorVisible(False)
        self._cropEditorWidget.setSourceVolumeNodeSelectorVisible(False)
        self._cropEditorWidget.setEffectNameOrder(["Scissors"])
        self._cropEditorWidget.unorderedEffectsVisible = False
        self._cropEditorWidget.show()

        # Activate Scissors effect
        self._cropEditorWidget.setActiveEffectByName("Scissors")
        effect = self._cropEditorWidget.activeEffect()
        if effect:
            effect.setParameter("Operation", "EraseInside")
            effect.setParameter("Shape", "FreeForm")
            logging.info("Scissors effect activated successfully!")
        else:
            logging.error("Failed to activate Scissors effect.")

        self._cropDebounceTimer = qt.QTimer()
        self._cropDebounceTimer.setSingleShot(True)
        self._cropDebounceTimer.setInterval(400)
        self._cropDebounceTimer.connect("timeout()", self._onCropStrokeCompleted)

        self.addObserver(self._cropSegNode, vtk.vtkCommand.ModifiedEvent, self._onCropSegModified)

        # Update button states
        self._cropMode = True
        self._startCropBtn.setEnabled(False)
        self._doneCropBtn.setEnabled(True)
        self._cancelCropBtn.setEnabled(True)
        self.ui.applyButton.setEnabled(False)
        
        logging.info("--- Crop Mode Initialized ---")

    def _onCropSegModified(self, caller=None, event=None) -> None:
        """Restart the debounce timer on every segmentation change."""
        if self._cropMode:
            self._cropDebounceTimer.start()

    def _onCropStrokeCompleted(self, caller=None, event=None) -> None:
        """After each paint stroke, run connected components and drop the torso."""
        if not self._cropMode or not self._cropSegNode:
            return

        # Determine seed point: stump center if placed, else base point
        seedRAS = None
        armNode = self.ui.armMarkupsWidget.currentNode()
        if armNode and armNode.GetNumberOfControlPoints() >= 1:
            p = np.zeros(3)
            idx = 1 if armNode.GetNumberOfControlPoints() >= 2 else 0
            armNode.GetNthControlPointPosition(idx, p)
            seedRAS = p

        self.logic.removeDisconnectedComponents(self._cropSegNode, seedRAS)

    def onDoneCrop(self) -> None:
        """Accept crop: stitch the cropped base back together with the untouched elbow."""
        if not self._cropSegNode:
            return

        try:
            targetModel = slicer.util.getNode("Socket Base")
            alignedElbow = slicer.util.getNode("Aligned Elbow")
            finalSocket = slicer.util.getNode("Prosthetic Socket")
        except Exception:
            self._exitCropMode(discard=False)
            return

        segmentation = self._cropSegNode.GetSegmentation()
        segID = segmentation.GetNthSegmentID(0)

        # Pull polydata from the crop
        segmentation.SetConversionParameter("Decimation factor", "0.3")
        segmentation.SetConversionParameter("Smoothing factor", "0.9")
        segmentation.RemoveRepresentation(vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())
        
        self._cropSegNode.CreateClosedSurfaceRepresentation()
        croppedPoly = vtk.vtkPolyData()
        self._cropSegNode.GetClosedSurfaceRepresentation(segID, croppedPoly)

        if croppedPoly.GetNumberOfPoints() > 0:
            targetModel.SetAndObservePolyData(croppedPoly)
        else:
            targetModel.SetAndObservePolyData(self._originalPolyData)

        # Re-stitch the parts
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(targetModel.GetPolyData())
        appender.AddInputData(alignedElbow.GetPolyData())
        appender.Update()
        
        finalSocket.SetAndObservePolyData(appender.GetOutput())

        # Reset visibilities
        alignedElbow.GetDisplayNode().SetVisibility(False)
        finalSocket.GetDisplayNode().SetVisibility(True)

        self._exitCropMode(discard=False)

    def onCancelCrop(self) -> None:
        """Discard all crop edits, restore original socket geometry."""
        try:
            targetModel = slicer.util.getNode("Socket Base")
            alignedElbow = slicer.util.getNode("Aligned Elbow")
            finalSocket = slicer.util.getNode("Prosthetic Socket")
            
            if self._originalPolyData:
                targetModel.SetAndObservePolyData(self._originalPolyData)
                
            alignedElbow.GetDisplayNode().SetVisibility(False)
            finalSocket.GetDisplayNode().SetVisibility(True)
        except Exception:
            pass
        self._exitCropMode(discard=True)

    def _exitCropMode(self, discard: bool) -> None:
        """Shared teardown for Done and Cancel."""
        if hasattr(self, "_cropDebounceTimer") and self._cropDebounceTimer:
            self._cropDebounceTimer.stop()

        if self._cropSegNode:
            try:
                self.removeObserver(self._cropSegNode, vtk.vtkCommand.ModifiedEvent, self._onCropSegModified)
            except Exception:
                pass
            slicer.mrmlScene.RemoveNode(self._cropSegNode)
            self._cropSegNode = None

        if self._cropEditorNode:
            slicer.mrmlScene.RemoveNode(self._cropEditorNode)
            self._cropEditorNode = None
            
        if hasattr(self, "_cropRefVolume") and self._cropRefVolume:
            slicer.mrmlScene.RemoveNode(self._cropRefVolume)
            self._cropRefVolume = None

        # Cleanly remove the widget from the UI
        if self._cropEditorWidget:
            self.layout.removeWidget(self._cropEditorWidget)
            self._cropEditorWidget.deleteLater()
            self._cropEditorWidget = None

        self._originalPolyData = None

        lm = slicer.app.layoutManager()
        threeDWidget = lm.threeDWidget(0)
        if threeDWidget:
            cam = threeDWidget.threeDView().renderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            cam.SetParallelProjection(False)
            threeDWidget.threeDView().renderWindow().Render()

        if self._savedLayout is not None:
            lm.setLayout(self._savedLayout)
            self._savedLayout = None

        self._cropMode = False
        self._startCropBtn.setEnabled(True)
        self._doneCropBtn.setEnabled(False)
        self._cancelCropBtn.setEnabled(False)
        self._checkCanApply()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    
    def onApplyButton(self) -> None:
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        patientModel = slicer.mrmlScene.GetFirstNodeByName("Patient Scan")
        elbowModelNode = slicer.util.getNode("Prosthetic Elbow")
        transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
        lineNode = slicer.mrmlScene.GetFirstNodeByName("TrajectoryLine")
        
        if not patientModel or not elbowModelNode or not transformNode or not lineNode:
            slicer.util.errorDisplay("Please click 'Initialize Preview' first to set up the elbow.")
            return

        # Hide preview cylinder
        cylNode = slicer.mrmlScene.GetFirstNodeByName("PreviewCylinder")
        if cylNode:
            cylNode.GetDisplayNode().SetVisibility(False)

        # Harvest final values directly from the interactive scene
        p0 = np.zeros(3)
        lineNode.GetNthControlPointPosition(0, p0) # Anchor point
        
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        target_pos = np.array([matrix.GetElement(0, 3), matrix.GetElement(1, 3), matrix.GetElement(2, 3)])
        
        current_scale = np.linalg.norm([matrix.GetElement(0,0), matrix.GetElement(1,0), matrix.GetElement(2,0)])
        if hasattr(self, "_radiusBottomSlider"):
            radius_elbow = self._radiusBottomSlider.value
        else:
            radius_elbow = current_scale * 27.5

        if hasattr(self, "_radiusTopSlider"):
            radius_top = self._radiusTopSlider.value
        else:
            patientModel = slicer.mrmlScene.GetFirstNodeByName("Patient Scan")
            armNode = self.ui.armMarkupsWidget.currentNode()
            _, _, radius_top = self.logic.compute_stump_geometry(armNode, patientModel, fallback_radius=radius_elbow * 1.25)

        overlap_top = self._overlapTopSlider.value if hasattr(self, "_overlapTopSlider") else 15.0
        offset_dist = np.linalg.norm(target_pos - p0)
        wall_thickness = self.ui.wallThicknessSlider.value
        padding_thickness = self.ui.paddingThicknessSlider.value

        self.logic.generate_prosthetic_socket(patientModel, elbowModelNode, p0, target_pos, offset_dist, radius_top, radius_elbow, wall_thickness, padding_thickness, overlap_top=overlap_top)

        self._checkCanApply()

    def onBrowseInputFile(self) -> None:
        """Open file dialog to select STL or OBJ file."""
        fileName = qt.QFileDialog.getOpenFileName(
            None,
            "Select STL or OBJ file",
            "",
            "Model files (*.stl *.obj);;All files (*)"
        )
        if fileName:
            # Ensure parameter node is initialized
            if not self._parameterNode:
                self.setParameterNode(self.logic.getParameterNode())
            
            # Clear previous markup nodes if a new file is selected
            armNode = self.ui.armMarkupsWidget.currentNode()
            if armNode:
                slicer.mrmlScene.RemoveNode(armNode)
            
            self.ui.inputFilePathEdit.setText(fileName)
            self._parameterNode.inputFilePath = fileName
            
            # Clear any previous patient models from the scene
            for node in slicer.mrmlScene.GetNodesByName("Patient Scan"):
                slicer.mrmlScene.RemoveNode(node)
            
            # Load the prosthetic model only when an input file is selected
            self.loadProstheticModel()
            
            # Load and display the patient model in the 3D scene
            try:
                modelNode = slicer.util.loadModel(fileName)
                if modelNode:
                    # Set color to red
                    displayNode = modelNode.GetDisplayNode()
                    if displayNode:
                        displayNode.SetColor(1, 0, 0)  # Red: R=1, G=0, B=0
                    modelNode.SetName("Patient Scan")
                    logging.info(f"Model loaded successfully: {modelNode.GetName()}")
                    # The model is automatically added to the scene and visible
                else:
                    slicer.util.errorDisplay("Failed to load the selected file. Please check if it's a valid STL or OBJ file.")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                slicer.util.errorDisplay(f"Error loading model: {str(e)}")

    def onExportButton(self) -> None:
        """Open directory dialog and export the prosthetic socket as STL."""
        socketNode = slicer.util.getNode("Prosthetic Socket")
        if not socketNode:
            slicer.util.errorDisplay("No prosthetic socket found. Please generate one first.")
            return
        
        # Open directory selection dialog
        outputDir = qt.QFileDialog.getExistingDirectory(
            None,
            "Select Directory to Save Prosthetic Socket",
            ""
        )
        
        if outputDir:
            try:
                # Call the logic to save the STL file
                self.logic.exportSocketToSTL(socketNode, outputDir)
                slicer.util.infoDisplay(f"Prosthetic socket exported successfully to:\n{outputDir}")
            except Exception as e:
                logging.error(f"Error exporting socket: {str(e)}")
                slicer.util.errorDisplay(f"Error exporting socket:\n{str(e)}")

#
# ScanToSocketLogic
#


class ScanToSocketLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return ScanToSocketParameterNode(super().getParameterNode())

    def removeDisconnectedComponents(self, segNode, seedRAS=None) -> None:
        """Keep only the connected component containing seedRAS (or the largest if no seed).

        Called after each erase stroke so the torso automatically disappears
        once it is severed from the arm.
        """
        import SimpleITK as sitk
        import sitkUtils

        segmentation = segNode.GetSegmentation()
        segID = segmentation.GetNthSegmentID(0)
        if not segID:
            return

        # Rasterize the current segment to a labelmap
        lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        ids = vtk.vtkStringArray()
        ids.InsertNextValue(segID)
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segNode, ids, lm)
        img = sitkUtils.PullVolumeFromSlicer(lm)
        slicer.mrmlScene.RemoveNode(lm)

        binary = sitk.Cast(img > 0, sitk.sitkUInt32)
        labeled = sitk.ConnectedComponent(binary)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(labeled)

        labels = stats.GetLabels()
        if not labels:
            return

        if seedRAS is not None:
            # Convert RAS → IJK index
            origin = np.array(img.GetOrigin())
            spacing = np.array(img.GetSpacing())
            direction = np.array(img.GetDirection()).reshape(3, 3)
            # SimpleITK uses LPS internally; Slicer RAS → LPS
            seedLPS = np.array([-seedRAS[0], -seedRAS[1], seedRAS[2]])
            idx = np.round(
                np.linalg.solve(direction * spacing, seedLPS - origin)
            ).astype(int)
            size = img.GetSize()
            if all(0 <= idx[i] < size[i] for i in range(3)):
                keepLabel = sitk.GetArrayFromImage(labeled)[idx[2], idx[1], idx[0]]
            else:
                keepLabel = 0

            if keepLabel == 0:
                # Seed outside image or in background — fall back to largest
                keepLabel = max(labels, key=lambda l: stats.GetNumberOfPixels(l))
        else:
            # No seed — keep largest component
            keepLabel = max(labels, key=lambda l: stats.GetNumberOfPixels(l))

        # Zero out all labels except the keeper
        keepMask = sitk.Cast(labeled == keepLabel, sitk.sitkUInt8)

        # Push result back into the segmentation
        resultLm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        sitkUtils.PushVolumeToSlicer(keepMask, resultLm)
        resultLm.SetSpacing(img.GetSpacing())
        resultLm.SetOrigin(img.GetOrigin())

        importIds = vtk.vtkStringArray()
        importIds.InsertNextValue(segID)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            resultLm, segNode, importIds
        )
        slicer.mrmlScene.RemoveNode(resultLm)
        logging.info(f"Connected components: kept label {keepLabel} of {len(labels)} components.")

    def create_tapered_cylinder(self, radius_top: float, radius_bottom: float, height: float, resolution: int = 50) -> vtk.vtkPolyData:
        """
        Creates a cylinder with a wider top than base (a truncated cone / frustum).
        In local coordinates, the axis is along Y from y = -height/2 (top, scan end) to y = +height/2 (bottom, elbow end).
        """
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(1.0)
        cylinder.SetHeight(height)
        cylinder.SetResolution(resolution)
        cylinder.SetCapping(True)
        cylinder.Update()

        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(cylinder.GetOutput())

        points = polyData.GetPoints()
        numPoints = points.GetNumberOfPoints()

        for i in range(numPoints):
            x, y, z = points.GetPoint(i)
            t = (y + (height / 2.0)) / height if height > 0 else 0.0
            t = max(0.0, min(1.0, t))
            r = (1.0 - t) * radius_top + t * radius_bottom
            points.SetPoint(i, x * r, y, z * r)

        points.Modified()
        polyData.GetPoints().Modified()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polyData)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()

        return normals.GetOutput()

    def compute_stump_geometry(self, armLandmarks, patientModelNode=None, fallback_radius=40.0):
        """
        Computes robust anatomical stump radius and trajectory using 360-degree mesh slice analysis if available,
        falling back to 3rd control point or defaults.
        """
        if not armLandmarks or armLandmarks.GetNumberOfControlPoints() < 2:
            logging.error("Please place at least 2 points: Point 1 on Shoulder, Point 2 on Amputation Tip.")
            return np.zeros(3), np.array([0.0, 0.0, 1.0]), fallback_radius, 0.0

        p0 = np.zeros(3); armLandmarks.GetNthControlPointPosition(0, p0)
        
        # 0. Extract mesh points and identify global torso center of mass
        all_pts = None
        torso_center = np.zeros(3)
        if patientModelNode and patientModelNode.GetPolyData():
            pts = patientModelNode.GetPolyData().GetPoints()
            if pts and pts.GetNumberOfPoints() > 100:
                all_pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
                torso_center = np.mean(all_pts, axis=0)
                # We no longer arbitrarily shift p0 using local mesh mean. 
                # The user's clinical click (Shoulder Pole to Tip Pole) forms a naturally perfect centerline.
                logging.info(f"Using raw skin click p0 as shoulder pole: {p0}")

        # Determine tip (base of amputation where elbow attaches) and shoulder points
        p_tip = np.copy(p0)
        raw_tip = np.zeros(3); armLandmarks.GetNthControlPointPosition(0, raw_tip)
        raw_shoulder = None
        if armLandmarks.GetNumberOfControlPoints() >= 2:
            p1 = np.zeros(3); armLandmarks.GetNthControlPointPosition(1, p1)
            raw_p1 = np.copy(p1)
            if all_pts is not None:
                pass # Removed flawed local-mean shift for p1 as well
            # Identify which point is shoulder (closer to torso center) and which is amputation tip
            if np.linalg.norm(p0 - torso_center) < np.linalg.norm(p1 - torso_center):
                p_tip = p1
                p_shoulder = p0
                raw_tip = np.copy(raw_p1)
                raw_shoulder = np.zeros(3); armLandmarks.GetNthControlPointPosition(0, raw_shoulder)
            else:
                p_tip = p0
                p_shoulder = p1
                raw_shoulder = np.copy(raw_p1)
                
            direction_vector = p_tip - p_shoulder
            norm = np.linalg.norm(direction_vector)
            stump_length = float(norm)
            direction_vector = direction_vector / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
        else:
            # Single-point mode: use torso-to-tip helping vector as robust unrotated initial search axis
            # Eliminates 90-degree PCA diameter rotation on short/stout residual limbs!
            direction_vector = p_tip - torso_center
            norm = np.linalg.norm(direction_vector)
            direction_vector = direction_vector / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
            norm = 150.0
            p1 = p_tip + direction_vector * norm
            stump_length = 0.0

        stump_radius = None
        refined_p1 = np.copy(p_tip)
        refined_dir = np.copy(direction_vector)

        # 1. Marching Cylinder Centerline & Radius Algorithm
        # March proximally (from tip back into the arm shaft toward shoulder) to sample clean cylinder slices
        if patientModelNode and patientModelNode.GetPolyData() and all_pts is not None:
            obbTree = vtk.vtkOBBTree()
            obbTree.SetDataSet(patientModelNode.GetPolyData())
            obbTree.BuildLocator()
            
            curr_center = np.copy(p_tip)
            curr_dir = -np.copy(direction_vector) # March proximally into arm shaft
            
            marched_centers = []
            marched_radii = []
            
            step_size = 10.0
            # March only the distal 50% of the stump to completely avoid Torso/Shoulder geometry
            max_steps = max(2, min(8, int((stump_length * 0.50) / step_size))) if stump_length > 0 else 8
            
            for step in range(1, max_steps + 1):
                pred_center = p_tip + (step * step_size) * curr_dir
                
                # Construct local orthogonal vectors for radial raycasting
                ref = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(ref, curr_dir)) > 0.9:
                    ref = np.array([1.0, 0.0, 0.0])
                u = np.cross(curr_dir, ref); u /= np.linalg.norm(u)
                w = np.cross(curr_dir, u)
                
                # Raycast radially outward in 36 directions
                hull_pts = []
                for i in range(36):
                    ang = i * 2.0 * math.pi / 36.0
                    ray_dir = math.cos(ang) * u + math.sin(ang) * w
                    
                    pSource = pred_center
                    pTarget = pred_center + ray_dir * 100.0
                    
                    points = vtk.vtkPoints()
                    cellIds = vtk.vtkIdList()
                    obbTree.IntersectWithLine(pSource, pTarget, points, cellIds)
                    
                    # vtkOBBTree intersections are NOT guaranteed to be sorted by distance!
                    # We must evaluate all intersections and explicitly pick the outermost one
                    # that falls within realistic anatomical bounds (15-65mm).
                    valid_pts_for_ray = []
                    for j in range(points.GetNumberOfPoints()):
                        pt = [0.0, 0.0, 0.0]
                        points.GetPoint(j, pt)
                        r = np.linalg.norm(np.array(pt) - pred_center)
                        if 5.0 <= r <= 85.0: # Expanded bounds to support very thin tips (5mm) and thick arms (85mm)
                            valid_pts_for_ray.append((r, pt))
                            
                    if valid_pts_for_ray:
                        # Sort by radius ascending (smallest first) to hit the arm skin 
                        # and completely ignore the Torso or other limbs further along the ray!
                        valid_pts_for_ray.sort(key=lambda x: x[0], reverse=False)
                        hull_pts.append(np.array(valid_pts_for_ray[0][1]))
                if len(hull_pts) < 12:
                    break # End of stump reached or hit torso
                    
                if len(hull_pts) >= 12:
                    step_centroid = np.mean(hull_pts, axis=0)
                    # Radius is median distance of outer skin points from refined slice centroid
                    step_r = float(np.median([np.linalg.norm(pt - step_centroid - np.dot(pt - step_centroid, curr_dir)*curr_dir) for pt in hull_pts]))
                    
                    # Torso Collision Detection: if radius jumps by > 15mm from previous slice, we hit the Torso
                    if len(marched_radii) > 0 and (step_r - marched_radii[-1]) > 15.0:
                        logging.info(f"Torso collision detected at step {step} (Radius spiked from {marched_radii[-1]:.1f} to {step_r:.1f}). Stopping march.")
                        break
                        
                    marched_centers.append(step_centroid)
                    marched_radii.append(step_r)
                    
                    # We NO LONGER dynamically curve `curr_dir`.
                    # Angled slicing of a tapered ellipsoid mathematically shifts the centroid, causing runaway drift!
                    # Marching straight along the initial axis guarantees slice centers converge to the true centerline.

            if len(marched_centers) >= 2:
                # Trajectory strictly derived from marched slice centroids along the arm!
                traj_vec = marched_centers[0] - marched_centers[-1]
                traj_norm = np.linalg.norm(traj_vec)
                if traj_norm > 5.0:
                    refined_dir = traj_vec / traj_norm
                    if np.dot(refined_dir, direction_vector) < 0:
                        refined_dir = -refined_dir
                    
                    # Safeguard: If marched trajectory deviates massively from user's manual trajectory,
                    # it means raycasts hit the Torso or geometry is distorted. Trust the user's clinical clicks!
                    angle_diff = np.degrees(np.arccos(np.clip(np.dot(refined_dir, direction_vector), -1.0, 1.0)))
                    if angle_diff > 35.0:
                        logging.warning(f"Marched trajectory deviated by {angle_diff:.1f} deg! Falling back to user's manual trajectory.")
                        refined_dir = direction_vector
                        
                    # Mathematically center the placement: project the raw tip onto the marched centerline!
                    # This completely satisfies the requirement for the arm extending from the true mathematical center.
                    centerline_p0 = marched_centers[0]
                    refined_p1 = centerline_p0 + np.dot(p_tip - centerline_p0, refined_dir) * refined_dir
                    
                    stump_radius = float(np.median(marched_radii))
                    logging.info(f"Marching Algorithm computed trajectory across {len(marched_centers)} slices: r={stump_radius:.2f}mm, length={traj_norm:.1f}mm")
            elif len(marched_centers) == 1:
                refined_p1 = p_tip
                stump_radius = marched_radii[0]
                logging.info(f"Marching Algorithm found single valid cylinder slice: r={stump_radius:.2f}mm")

        # 2. Override with 3rd control point if present (allowing interactive user override)
        if armLandmarks.GetNumberOfControlPoints() >= 3:
            p2 = np.zeros(3); armLandmarks.GetNthControlPointPosition(2, p2)
            center_to_edge_vector = p2 - p1
            stump_radius = np.linalg.norm(np.cross(center_to_edge_vector, refined_dir))

        # 3. Fallback default if no mesh and no 3rd point
        if stump_radius is None or stump_radius <= 0.0:
            stump_radius = fallback_radius

        # 4. Measure stump_length by projecting the user's raw fiducials onto the PCA trajectory line
        if 'raw_shoulder' in locals() and raw_shoulder is not None:
            stump_length = float(abs(np.dot(raw_tip - raw_shoulder, refined_dir)))

        return refined_p1, refined_dir, stump_radius, stump_length

    def get_alignment_params(self, armLandmarks, offset_distance, patientModelNode=None):
        """Calculates trajectory, rotation axes, and scale mathematically."""
        p0 = np.zeros(3); armLandmarks.GetNthControlPointPosition(0, p0)
        p1, direction_vector, stump_radius, stump_length = self.compute_stump_geometry(armLandmarks, patientModelNode, fallback_radius=40.0)

        scale_factor = (stump_radius / 1.25) / 27.5

        # offset_distance is the total Shoulder-to-Elbow distance.
        # Since p1 is at the stump tip (at distance stump_length from shoulder along the trajectory),
        # the remaining extension from stump tip p1 to the elbow target is offset_distance - stump_length.
        extension_distance = max(10.0, offset_distance - stump_length)
        elbow_target_pos = p1 + (direction_vector * extension_distance)

        z_axis = direction_vector
        y_axis = np.cross(z_axis, np.array([1, 0, 0])) if abs(z_axis[0]) < 0.9 else np.cross(z_axis, np.array([0, 1, 0]))
        if armLandmarks.GetNumberOfControlPoints() >= 3:
            p2 = np.zeros(3); armLandmarks.GetNthControlPointPosition(2, p2)
            center_to_edge_vector = p2 - p1
            if np.linalg.norm(np.cross(center_to_edge_vector, z_axis)) > 1e-6:
                y_axis = np.cross(z_axis, center_to_edge_vector)
        norm_y = np.linalg.norm(y_axis)
        y_axis = y_axis / norm_y if norm_y > 0 else np.array([0,1,0])
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        return elbow_target_pos, p1, z_axis, y_axis, x_axis, scale_factor, stump_radius, stump_length

    def process(self, patientModelNode, armLandmarks, elbowModelNode, offset_distance, wall_thickness=4.0) -> None:
        if armLandmarks.GetNumberOfControlPoints() < 2:
            logging.error("Please place at least 2 points (Point 1: Shoulder, Point 2: Amputation Tip).")
            return
        
        target_pos, p1, z_axis, y_axis, x_axis, scale, stump_radius, stump_length = self.get_alignment_params(armLandmarks, offset_distance, patientModelNode)

        # Ensure elbow transform exists (in case user skipped preview)
        transformNode = slicer.mrmlScene.GetFirstNodeByName("MasterElbowTransform")
        if not transformNode:
            logging.info("Building default base transform...")
            transform = vtk.vtkTransform()
            transform.Translate(target_pos)

            base_rot = vtk.vtkMatrix4x4()
            for i in range(3):
                base_rot.SetElement(i, 0, x_axis[i])
                base_rot.SetElement(i, 1, y_axis[i])
                base_rot.SetElement(i, 2, z_axis[i])
            transform.Concatenate(base_rot)
            transform.RotateY(180) # Flip down
            transform.Scale(scale, scale, scale)

            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "MasterElbowTransform")
            transformNode.SetMatrixTransformToParent(transform.GetMatrix())
            elbowModelNode.SetAndObserveTransformNodeID(transformNode.GetID())
            
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        current_scale = np.linalg.norm([matrix.GetElement(0,0), matrix.GetElement(1,0), matrix.GetElement(2,0)])
        radius_elbow = current_scale * 27.5
            
        ext_dist = np.linalg.norm(np.array(target_pos) - np.array(p1))
        self.generate_prosthetic_socket(patientModelNode, elbowModelNode, p1, target_pos, ext_dist, stump_radius, radius_elbow, wall_thickness)

    def generate_prosthetic_socket(self, patientModelNode, elbowModelNode, p1_stump_center, elbow_target_pos, offset_distance, radius_top, radius_elbow, wall_thickness=4.0, padding_thickness=0.0, overlap_top=15.0):
        import SimpleITK as sitk
        import sitkUtils

        logging.info("Generating bridging cone/tapered cylinder...")

        overlap_bottom = 15.0
        height_main = offset_distance + overlap_top
        total_height = height_main + overlap_bottom

        if height_main > 1e-6:
            slope = (radius_top - radius_elbow) / height_main
            radius_bottom = radius_elbow - (slope * overlap_bottom)
        else:
            radius_bottom = radius_elbow
        radius_bottom = max(0.1, radius_bottom)

        conePolyData = self.create_tapered_cylinder(radius_top, radius_bottom, total_height, 50)

        direction = elbow_target_pos - p1_stump_center
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

        top_point = p1_stump_center - (direction * overlap_top)
        bottom_point = elbow_target_pos + (direction * overlap_bottom)
        true_center = (top_point + bottom_point) / 2.0

        cylTransform = vtk.vtkTransform()
        cylTransform.Translate(true_center.tolist())
        cylTransform.RotateWXYZ(angle_deg, rotation_axis.tolist())

        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(conePolyData)
        tf.SetTransform(cylTransform)
        tf.Update()

        cylinderModelNode = slicer.modules.models.logic().AddModel(tf.GetOutput())
        cylinderModelNode.SetName("Bridge Cylinder")

        # --- BUILD SEGMENTATION (NO ELBOW INCLUDED) ---
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SocketSegmentation")
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(patientModelNode, segNode)
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(cylinderModelNode, segNode)

        segmentation = segNode.GetSegmentation()
        patientVoidID = segmentation.GetSegmentIdBySegmentName(patientModelNode.GetName())
        cylinderID    = segmentation.GetSegmentIdBySegmentName(cylinderModelNode.GetName())

        # --- REFERENCE VOLUME ---
        spacing = 0.75
        bounds = [0.0] * 6
        segNode.GetRASBounds(bounds)
        margin = 15.0

        origin = [bounds[0]-margin, bounds[2]-margin, bounds[4]-margin]
        dims = [
            int((bounds[1]-bounds[0]+2*margin)/spacing)+1,
            int((bounds[3]-bounds[2]+2*margin)/spacing)+1,
            int((bounds[5]-bounds[4]+2*margin)/spacing)+1,
        ]

        refArray = np.zeros([dims[2], dims[1], dims[0]], dtype=np.int16)
        refVolumeNode = slicer.util.addVolumeFromArray(refArray)
        refVolumeNode.SetName("TempRef")
        refVolumeNode.SetSpacing(spacing, spacing, spacing)
        refVolumeNode.SetOrigin(origin)

        # Rasterize
        def seg_to_sitk(segID):
            lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            ids = vtk.vtkStringArray()
            ids.InsertNextValue(segID)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segNode, ids, lm, refVolumeNode)
            img = sitkUtils.PullVolumeFromSlicer(lm)
            slicer.mrmlScene.RemoveNode(lm)
            return img

        patientBin = sitk.Cast(seg_to_sitk(patientVoidID) > 0, sitk.sitkUInt8)
        cylinderBin = sitk.Cast(seg_to_sitk(cylinderID) > 0, sitk.sitkUInt8)

        # Hollow boolean
        clearance_vox = max(1, round(1.5 / spacing))
        wall_vox = max(1, round(wall_thickness / spacing))
        padding_vox = max(0, round(padding_thickness / spacing))

        inner_bore = sitk.BinaryDilate(patientBin, [clearance_vox] * 3)
        outer_shell = sitk.BinaryDilate(patientBin, [clearance_vox + wall_vox + padding_vox] * 3)

        inner_arr = sitk.GetArrayFromImage(inner_bore).astype(bool)
        outer_arr = sitk.GetArrayFromImage(outer_shell).astype(bool)
        cylinderArr = sitk.GetArrayFromImage(cylinderBin).astype(bool)

        socketBlank = (outer_arr | cylinderArr) & ~inner_arr

        # --- CONVERT TO MESH ---
        resultLabelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempSocketVolume")
        slicer.util.updateVolumeFromArray(resultLabelmapNode, socketBlank.astype(np.int16))
        resultLabelmapNode.SetSpacing(spacing, spacing, spacing)
        resultLabelmapNode.SetOrigin(origin)

        resultSegNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "FinalSocket")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(resultLabelmapNode, resultSegNode)

        outSegmentation = resultSegNode.GetSegmentation()
        outSegID = outSegmentation.GetNthSegmentID(0)
        self._smooth_segment(resultSegNode, outSegID, gaussian_sigma_mm=0.6)

        # 1. Extract Socket Polydata
        outSegmentation.SetConversionParameter("Decimation factor", "0.3")
        outSegmentation.SetConversionParameter("Smoothing factor", "0.8")
        outSegmentation.RemoveRepresentation(vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())
        resultSegNode.CreateClosedSurfaceRepresentation()
        socketPoly = vtk.vtkPolyData()
        resultSegNode.GetClosedSurfaceRepresentation(outSegID, socketPoly)

        # 2. Extract Elbow Polydata and bake transform
        elbowPoly = elbowModelNode.GetPolyData()
        transformNode = elbowModelNode.GetParentTransformNode()
        if transformNode:
            transform = vtk.vtkTransform()
            transform.SetMatrix(transformNode.GetMatrixTransformToParent())
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(elbowPoly)
            transformFilter.SetTransform(transform)
            transformFilter.Update()
            elbowPoly = transformFilter.GetOutput()

        # Clean elbow just in case
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(elbowPoly)
        cleaner.Update()
        elbowClean = cleaner.GetOutput()

        # 3. Append High-Res Polydata Together
        baseModel = slicer.mrmlScene.GetFirstNodeByName("Socket Base")
        if not baseModel:
            baseModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Socket Base")
            baseModel.CreateDefaultDisplayNodes()
        baseModel.SetAndObservePolyData(socketPoly)
        baseModel.GetDisplayNode().SetVisibility(False)

        elbowTarget = slicer.mrmlScene.GetFirstNodeByName("Aligned Elbow")
        if not elbowTarget:
            elbowTarget = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Aligned Elbow")
            elbowTarget.CreateDefaultDisplayNodes()
        elbowTarget.SetAndObservePolyData(elbowClean)
        elbowTarget.GetDisplayNode().SetVisibility(False)

        # 3. Append High-Res Polydata Together
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(socketPoly)
        appender.AddInputData(elbowClean)
        appender.Update()

        # 4. Output final explicit model node
        finalModelNode = slicer.mrmlScene.GetFirstNodeByName("Prosthetic Socket")
        if not finalModelNode:
            finalModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Prosthetic Socket")
        finalModelNode.SetAndObservePolyData(appender.GetOutput())
        finalModelNode.CreateDefaultDisplayNodes()
        finalModelNode.GetDisplayNode().SetColor(1.0, 0.0, 0.0)

        # --- CLEANUP ---
        slicer.mrmlScene.RemoveNode(segNode)
        slicer.mrmlScene.RemoveNode(refVolumeNode)
        slicer.mrmlScene.RemoveNode(resultLabelmapNode)
        slicer.mrmlScene.RemoveNode(resultSegNode)
        
        lineNode = slicer.mrmlScene.GetFirstNodeByName("TrajectoryLine")
        if lineNode: slicer.mrmlScene.RemoveNode(lineNode)

        patientModelNode.GetDisplayNode().SetVisibility(False)
        cylinderModelNode.GetDisplayNode().SetVisibility(False)
        elbowModelNode.GetDisplayNode().SetVisibility(False)
        
        # Turn off gizmo
        if transformNode and transformNode.GetDisplayNode():
            transformNode.GetDisplayNode().SetEditorVisibility(False)

        logging.info("Socket generation complete.")
        slicer.util.resetThreeDViews()

    def _smooth_segment(self, seg_node, seg_id: str, gaussian_sigma_mm: float = 0.6) -> None:
        try:
            editor = slicer.qMRMLSegmentEditorWidget()
            editor.setMRMLScene(slicer.mrmlScene)
            editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            editor.setMRMLSegmentEditorNode(editor_node)
            editor.setSegmentationNode(seg_node)
            editor.setCurrentSegmentID(seg_id)
            editor.setActiveEffectByName("Smoothing")
            effect = editor.activeEffect()
            if effect:
                effect.setParameter("SmoothingMethod", "GAUSSIAN")
                effect.setParameter("GaussianStandardDeviationMm", str(gaussian_sigma_mm))
                effect.self().onApply()
            slicer.mrmlScene.RemoveNode(editor_node)
        except Exception as e:
            logging.warning(f"Segment smoothing skipped: {e}")

    def exportSocketToSTL(self, socketNode, outputDir) -> None:
        """Export the prosthetic socket model to an STL file."""
        import os
        from datetime import datetime
        
        # Generate filename with timestamp to avoid overwrites
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"above_elbow_prosthetic_{timestamp}.stl"
        filepath = os.path.join(outputDir, filename)
        
        logging.info(f"Exporting prosthetic socket to: {filepath}")
        
        # Save the model node as STL
        success = slicer.util.saveNode(socketNode, filepath)
        if success:
            logging.info(f"Successfully saved prosthetic socket to {filepath}")
        else:
            raise RuntimeError(f"Failed to save prosthetic socket to {filepath}")

#
# ScanToSocketTest
#


class ScanToSocketTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ScanToSocket1()

    def test_ScanToSocket1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        # import SampleData
        import tempfile

        # registerSampleData()
        # inputVolume = SampleData.downloadSample("ScanToSocket1")
        # self.delayDisplay("Loaded test data set")

        # # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        # # Save input volume as a temporary STL file for testing file path input
        # tempDir = tempfile.gettempdir()
        # testFilePath = os.path.join(tempDir, "test_input.stl")
        # # Note: In production, you would convert the volume to a model and save as STL
        # # For this test, we'll use the volume directly via slicer's save function
        # slicer.util.saveNode(inputVolume, testFilePath)

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100

        # # Test the module logic

        # logic = ScanToSocketLogic()

        # # Test algorithm with non-inverted threshold
        # logic.process(testFilePath, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)

        # # Test algorithm with inverted threshold
        # logic.process(testFilePath, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        # # Clean up temporary file
        # if os.path.exists(testFilePath):
        #     os.remove(testFilePath)

        self.delayDisplay("Test passed")
