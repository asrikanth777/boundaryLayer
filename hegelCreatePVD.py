# /usr/bin/python3
import glob
import numpy
import pandas
import xml.etree.ElementTree as ET

#### This file create a pvd file connecting together all the pvts file for the multiblock simulation of hegel


#-----------------------------------------------------------------------------------------------------------
# parameters    --- modify if necessary
rep        = './output (copy)/'
multiblock = 8                           # number of hegel mesh block division (min = 1)
parallel   = True                       # True -- is the hegel simulation parallel   ?
fluid      = 'LTE'                       # Hegel Fluid class (PF = "")
timestep   = 1.e-8 * 1e4                       #      -- used timestep : check the input file
steady     = False
# end parameter --- user should not have to modify anything below this part
#-----------------------------------------------------------------------------------------------------------

pvd_file = './hegel.pvd'
if (parallel) :
    suffix = ".pvts"
else :
    suffix = ".vts"
files_ = numpy.array(glob.glob(rep + "flowfield" + "*" + suffix))

if (not steady) :
    # sort the file !
    idx = numpy.array([])
    loc = numpy.array([])
    num = 0
    for ii in files_ :
        if (multiblock > 1) :
            val = ii.split("flowfield_NS_"+fluid+"_")[1].split("_B")
            if len(val) == 2:
                if (float(val[0]) % 1000 == 0):
                    idx  = numpy.append(idx,float(val[0]))
                    loc = numpy.append(loc,num)
            elif len(val) == 1:
                idx  = numpy.append(idx, 0)
                loc = numpy.append(loc,num)
            num += 1
        else :
            idx  = numpy.append(idx, float(ii.split("flowfield_NS_"+fluid+"_")[1].split(".")[0]))
    sort = numpy.argsort(idx)
    loc = loc[sort]
loc = loc.astype(int)
#print(loc)      
# create the pvd file
vtkFile    = ET.Element("VTKFile", type="Collection", version="0.1")   
collection = ET.SubElement(vtkFile, "Collection")   

#print(sort.size)

s = 0 ; ss = 0 ;
for ii in range (sort.size) :
    if (s%multiblock) == 0 :
        ss += 1
        # print("ss = ", ss)
    if (not steady) :
        if (multiblock > 1) :
            # print(files_[sort[ii]])
            block   = files_[loc[ii]].split("_B")[1].split(".")[0]
            # print(block)
            dataset = ET.SubElement(collection, "DataSet", timestep="%8.8e" %((ss)*timestep), part=block, file=files_[loc[ii]])
        else :
            dataset = ET.SubElement(collection, "DataSet", timestep="%8.8e" %(ss*timestep), part='0', file=files_[loc[ii]].split(rep)[1])
    else :
        if (multiblock > 1) :
            dataset = ET.SubElement(collection, "DataSet", timestep="%8.8e" %(0.), part=str(ii), file=files_[ii])
        else :
            dataset = ET.SubElement(collection, "DataSet", timestep="%8.8e" %(0.), part='0', file=files_[ii])
    s+=1
             
from xml.dom import minidom
import os 
# Print pretty
xmlstr = minidom.parseString(ET.tostring(vtkFile)).toprettyxml(indent="")
with open(pvd_file, "w") as f:
    f.write(xmlstr)
