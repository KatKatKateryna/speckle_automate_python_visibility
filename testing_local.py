
import math
from typing import List
from specklepy.api.operations import receive, send
from specklepy.objects.geometry import Line, Point, Pointcloud 
from specklepy.objects.other import Collection

import numpy as np 
from operator import add, sub 
import matplotlib as mpl

from flatten import iterateBase
from utils.getComment import get_comments

from utils.utils_other import RESULT_BRANCH, cleanPtsList, findMeshesNearby, sortPtsByMesh
from utils.utils_visibility import getAllPlanes, projectToPolygon, rotate_vector, expandPtsList

HALF_VIEW_DEGREES = 70
STEP_DEGREES = 1
STEP_DEGREES_EXTENSION = 10

def run(client, server_transport, keyword):
    
    onlyIllustrate = False  

    project_id = server_transport.stream_id

    comments = get_comments(
        client,
        project_id,
    )
    pt_origin = None
    dir = None
    commitId = None
    for item in comments["comments"]["items"]:
        if keyword.lower() in item["rawText"].lower():

            viewerState = item["viewerState"]
            commitId = viewerState["resources"]["request"]["resourceIdString"].split("@")[1]
            pt_origin: List[float] = viewerState["ui"]["selection"]

            pos: List[float] = viewerState["ui"]["camera"]["position"]
            target: List[float] = viewerState["ui"]["camera"]["target"]

            dir = list( map(sub, pos, target) )
            break 
    if pt_origin is None or commitId is None: 
        return 

    commit = client.commit.get(project_id, commitId)
    base = receive(commit.referencedObject, server_transport)
    objects = [b for b in iterateBase(base)]

    lines = []
    cloud = []
    dir = np.array(dir)
    start = Point.from_list(pt_origin)
    vectors = rotate_vector(pt_origin, dir, HALF_VIEW_DEGREES, STEP_DEGREES)
    #endPt = list( map(add,pt_origin,dir) )

    # just to find the line
    if onlyIllustrate is True:
        line = Line(start = start, end = Point.from_list(list( map(add,pt_origin,dir) )))
        line.units = "m"
        lines.append(line)
        for v in vectors:
            line = Line(start = start, end = Point.from_list( [v[0], v[1], v[2]]))
            line.units = "m"
            lines.append(line)
    
    ###########################
    else:
        usedVectors = {}
        all_pts = []
        count = 0
        all_geom = []
        for i, bld in enumerate(objects):
            # get all intersection points 
            meshes = getAllPlanes(bld)
            for mesh in meshes:
                if len(mesh)<3: continue 
                all_geom.append(mesh)
                pts, usedVectors = projectToPolygon(pt_origin, vectors, usedVectors, mesh, count) #Mesh.create(vertices = [0,0,0,5,0,0,5,19,0,0,14,0], faces=[4,0,1,2,3]))
                all_pts.extend(pts)
                count +=1

        cleanPts = cleanPtsList(pt_origin, all_pts, usedVectors)

        ### expand number of pts around filtered rays 
        expandedPts2 = []
        mesh_nearby = findMeshesNearby(cleanPts)
        expandedPts21, usedVectors2 = expandPtsList(pt_origin, cleanPts, {}, STEP_DEGREES_EXTENSION, all_geom, mesh_nearby)
        expandedPts2 = cleanPtsList(pt_origin, expandedPts21, usedVectors2)

        ### expand number of pts around filtered rays 
        expandedPts3 = []
        clean_extended_pts = cleanPts + expandedPts2
        mesh_nearby = findMeshesNearby(clean_extended_pts)
        expandedPts31, usedVectors3 = expandPtsList(pt_origin, clean_extended_pts, {}, STEP_DEGREES_EXTENSION/5, all_geom, mesh_nearby)
        expandedPts3 = cleanPtsList(pt_origin, expandedPts31, usedVectors3)

        ### expand number of pts around filtered rays 
        expandedPts4 = []
        #clean_extended_pts = clean_extended_pts + expandedPts3
        #mesh_nearby = findMeshesNearby(clean_extended_pts)
        #expandedPts4, usedVectors4 = expandPtsList(pt_origin, clean_extended_pts, {}, STEP_DEGREES/5, all_geom, mesh_nearby)

        sortedPts = sortPtsByMesh(cleanPts + expandedPts2 + expandedPts3 + expandedPts4)

        points = []
        colors = []
        distances = []

        for ptList in sortedPts:
            for p in ptList:
                points.extend([p.x, p.y, p.z])
                distances.append(p.distance)

        for d in distances:
            fraction = math.pow( (max(distances)-d)/max(distances), 0.4 )
            # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            cmap = mpl.colormaps['jet']
            mapColor = cmap(fraction)
            r = int(mapColor[0]*255) # int(poly.count / maxCount)*255
            g = int(mapColor[1]*255) # int(poly.count / maxCount)*255
            b = int(mapColor[2]*255) # 255 - int( poly.count / maxCount)*255

            color = (255<<24) + (r<<16) + (g<<8) + b # argb
            colors.append(color)
        if len(points)==0 or len(colors)==0: return  

        visibility = (len(vectors) - len(cleanPts))/len(vectors) 
        print(f"Visible sky: {visibility * 100}%")      

        cloud = [ Pointcloud(points = points, colors = colors, visibility = visibility )]

    
    if onlyIllustrate is True:
        visibleObj = Collection(elements = lines, units = "m", name = "Context", collectionType = "VisibilityAnalysis")
    else:
        visibleObj = Collection(elements = cloud, units = "m", name = "Context", collectionType = "VisibilityAnalysis")
    
    # create branch if needed 
    existing_branch = client.branch.get(project_id, RESULT_BRANCH, 1)  
    if existing_branch is None: 
        br_id = client.branch.create(stream_id = project_id, name = RESULT_BRANCH, description = "") 

    objId = send(visibleObj, transports=[server_transport]) 
    commit_id = client.commit.create(
                stream_id=project_id,
                object_id=objId,
                branch_name=RESULT_BRANCH,
                message="Automate Pointcloud",
                source_application="Python",
            )

r'''
# FOR DEBUGGING LOCALLY run this file
from specklepy.transports.server import ServerTransport
from specklepy.api.client import SpeckleClient
from specklepy.api.credentials import get_local_accounts

KEYWORD = "window"
server_url = "https://latest.speckle.dev/" #"https://speckle.xyz/" # project_data.speckle_server_url
project_id = "4ea6a03993" #"4ea6a03993"# Kate's tests #"17b0b76d13" #project_data.project_id
account = get_local_accounts()[0]
client = SpeckleClient(server_url)
client.authenticate_with_token(account.token)
server_transport = ServerTransport(project_id, client)
run(client, server_transport, KEYWORD) 
''' 
