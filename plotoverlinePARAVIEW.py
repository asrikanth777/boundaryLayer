# trace generated using paraview version 5.11.2
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
groupDatasets1 = FindSource('GroupDatasets1')

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=groupDatasets1)
plotOverLine1.Point1 = [0.401885, -1.89703e-19, 0.0]
plotOverLine1.Point2 = [0.712482, 0.192823, 0.0]

# find source
xMLStructuredGridReader1 = FindSource('XMLStructuredGridReader1')

# find source
xMLStructuredGridReader2 = FindSource('XMLStructuredGridReader2')

# find source
xMLStructuredGridReader3 = FindSource('XMLStructuredGridReader3')

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
plotOverLine1Display.Representation = 'Surface'
plotOverLine1Display.ColorArrayName = [None, '']
plotOverLine1Display.SelectTCoordArray = 'None'
plotOverLine1Display.SelectNormalArray = 'None'
plotOverLine1Display.SelectTangentArray = 'None'
plotOverLine1Display.OSPRayScaleArray = 'H'
plotOverLine1Display.OSPRayScaleFunction = 'PiecewiseFunction'
plotOverLine1Display.SelectOrientationVectors = 'None'
plotOverLine1Display.ScaleFactor = 0.03105969727039337
plotOverLine1Display.SelectScaleArray = 'H'
plotOverLine1Display.GlyphType = 'Arrow'
plotOverLine1Display.GlyphTableIndexArray = 'H'
plotOverLine1Display.GaussianRadius = 0.0015529848635196686
plotOverLine1Display.SetScaleArray = ['POINTS', 'H']
plotOverLine1Display.ScaleTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.OpacityArray = ['POINTS', 'H']
plotOverLine1Display.OpacityTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.DataAxesGrid = 'GridAxesRepresentation'
plotOverLine1Display.PolarAxes = 'PolarAxesRepresentation'
plotOverLine1Display.SelectInputVectors = [None, '']
plotOverLine1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
plotOverLine1Display.OSPRayScaleFunction.Points = [-7.947353839874268, 0.0, 0.5, 0.0, 14.630976676940918, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine1Display.ScaleTransferFunction.Points = [2063900.0, 0.0, 0.5, 0.0, 57955400.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine1Display.OpacityTransferFunction.Points = [2063900.0, 0.0, 0.5, 0.0, 57955400.0, 1.0, 0.5, 0.0]

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 'XYChartRepresentation')

# trace defaults for the display properties.
plotOverLine1Display_1.UseIndexForXAxis = 0
plotOverLine1Display_1.XArrayName = 'arc_length'
plotOverLine1Display_1.SeriesVisibility = ['H', 'M', 'p', 'rho', 'T', 'v_Magnitude']
plotOverLine1Display_1.SeriesLabel = ['arc_length', 'arc_length', 'H', 'H', 'M', 'M', 'p', 'p', 'rho', 'rho', 'T', 'T', 'v_X', 'v_X', 'v_Y', 'v_Y', 'v_Magnitude', 'v_Magnitude', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', 'H', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'M', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'p', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'rho', '0.6', '0.3100022888532845', '0.6399938963912413', 'T', '1', '0.5000076295109483', '0', 'v_X', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867', 'v_Y', '0', '0', '0', 'v_Magnitude', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'vtkValidPointMask', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'Points_X', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'Points_Y', '0.6', '0.3100022888532845', '0.6399938963912413', 'Points_Z', '1', '0.5000076295109483', '0', 'Points_Magnitude', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867']
plotOverLine1Display_1.SeriesOpacity = ['arc_length', '1.0', 'H', '1.0', 'M', '1.0', 'p', '1.0', 'rho', '1.0', 'T', '1.0', 'v_X', '1.0', 'v_Y', '1.0', 'v_Magnitude', '1.0', 'vtkValidPointMask', '1.0', 'Points_X', '1.0', 'Points_Y', '1.0', 'Points_Z', '1.0', 'Points_Magnitude', '1.0']
plotOverLine1Display_1.SeriesPlotCorner = ['arc_length', '0', 'H', '0', 'M', '0', 'p', '0', 'rho', '0', 'T', '0', 'v_X', '0', 'v_Y', '0', 'v_Magnitude', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesLabelPrefix = ''
plotOverLine1Display_1.SeriesLineStyle = ['arc_length', '1', 'H', '1', 'M', '1', 'p', '1', 'rho', '1', 'T', '1', 'v_X', '1', 'v_Y', '1', 'v_Magnitude', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesLineThickness = ['arc_length', '2', 'H', '2', 'M', '2', 'p', '2', 'rho', '2', 'T', '2', 'v_X', '2', 'v_Y', '2', 'v_Magnitude', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['arc_length', '0', 'H', '0', 'M', '0', 'p', '0', 'rho', '0', 'T', '0', 'v_X', '0', 'v_Y', '0', 'v_Magnitude', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesMarkerSize = ['arc_length', '4', 'H', '4', 'M', '4', 'p', '4', 'rho', '4', 'T', '4', 'v_X', '4', 'v_Y', '4', 'v_Magnitude', '4', 'vtkValidPointMask', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'Points_Magnitude', '4']

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesOpacity = ['arc_length', '1', 'H', '1', 'M', '1', 'p', '1', 'rho', '1', 'T', '1', 'v_X', '1', 'v_Y', '1', 'v_Magnitude', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesPlotCorner = ['H', '0', 'M', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'T', '0', 'arc_length', '0', 'p', '0', 'rho', '0', 'v_Magnitude', '0', 'v_X', '0', 'v_Y', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesLineStyle = ['H', '1', 'M', '1', 'Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'T', '1', 'arc_length', '1', 'p', '1', 'rho', '1', 'v_Magnitude', '1', 'v_X', '1', 'v_Y', '1', 'vtkValidPointMask', '1']
plotOverLine1Display_1.SeriesLineThickness = ['H', '2', 'M', '2', 'Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'T', '2', 'arc_length', '2', 'p', '2', 'rho', '2', 'v_Magnitude', '2', 'v_X', '2', 'v_Y', '2', 'vtkValidPointMask', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['H', '0', 'M', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'T', '0', 'arc_length', '0', 'p', '0', 'rho', '0', 'v_Magnitude', '0', 'v_X', '0', 'v_Y', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesMarkerSize = ['H', '4', 'M', '4', 'Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'T', '4', 'arc_length', '4', 'p', '4', 'rho', '4', 'v_Magnitude', '4', 'v_X', '4', 'v_Y', '4', 'vtkValidPointMask', '4']

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [0.401885, 0.0, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# Properties modified on plotOverLine1
plotOverLine1.Point2 = [0.712482, 0.0, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# Properties modified on plotOverLine1
plotOverLine1.Point2 = [0.6, 0.0, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [0.4, 0.0, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['M', 'p', 'rho', 'T', 'v_Magnitude']

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['p', 'rho', 'T', 'v_Magnitude']

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['p', 'rho', 'v_Magnitude']

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['rho', 'v_Magnitude']

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesVisibility = ['v_Magnitude']

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1387, 490)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [0.5569850889648221, 0.09229469144924668, 0.660900863442037]
renderView1.CameraFocalPoint = [0.556985088964822, 0.09229469144924665, -0.04535193596171183]
renderView1.CameraParallelScale = 0.1827916751426604

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).