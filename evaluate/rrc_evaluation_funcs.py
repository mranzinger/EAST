import json
import sys;sys.path.append('./')
import zipfile
import re
import sys
import os
import codecs
import importlib
from io import StringIO

import numpy as np

def print_help():
    sys.stdout.write('Usage: python %s.py -g=<gtFile> -s=<submFile> [-o=<outputFolder> -p=<jsonParams>]' %sys.argv[0])
    sys.exit(2)


def load_zip_file_keys(file,fileNameRegExp=''):
    """
    Returns an array with the entries of the ZIP file that match with the regular expression.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    """
    try:
        archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except :
        raise Exception('Error loading the ZIP archive.')

    pairs = []

    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)

        if addFile:
            pairs.append( keyName )

    return pairs


def load_zip_file(file,fileNameRegExp='',allEntries=False):
    """
    Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    allEntries validates that all entries in the ZIP file pass the fileNameRegExp
    """
    try:
        archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except :
        raise Exception('Error loading the ZIP archive')

    pairs = []
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)

        if addFile:
            pairs.append( [ keyName , archive.read(name)] )
        else:
            if allEntries:
                raise Exception('ZIP entry not valid: %s' %name)

    return dict(pairs)

def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        raw = codecs.decode(raw,'utf-8-sig', 'replace')
        #extracts BOM if exists
        raw = raw.encode('utf8')
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw.replace(codecs.BOM_UTF8, '', 1)
        return raw.decode('utf-8')
    except:
       return None

def validate_lines_in_file(fileName,file_contents,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    This function validates that all lines of the file calling the Line validation function for each line
    """
    utf8File = decode_utf8(file_contents)
    if (utf8File is None) :
        raise Exception("The file %s is not UTF-8" %fileName)

    lines = utf8File.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            try:
                validate_tl_line(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)
            except Exception as e:
                raise Exception(("Line in sample not valid. Sample: %s Line: %s Error: %s" %(fileName,line,str(e))).encode('utf-8', 'replace'))



def validate_tl_line(line,LTRB=True,withTranscription=True,withConfidence=True,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    """
    get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)


def get_tl_line_values(line,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    Returns values from a textline. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = "";
    points = []

    numPoints = 4;

    if LTRB:

        numPoints = 4;

        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence,transcription")
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence")
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,transcription")
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax")

        xmin = int(m.group(1))
        ymin = int(m.group(2))
        xmax = int(m.group(3))
        ymax = int(m.group(4))
        if(xmax<xmin):
                raise Exception("Xmax value (%s) not valid (Xmax < Xmin)." %(xmax))
        if(ymax<ymin):
                raise Exception("Ymax value (%s)  not valid (Ymax < Ymin)." %(ymax))

        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]

        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(xmin,ymin,imWidth,imHeight);
            validate_point_inside_bounds(xmax,ymax,imWidth,imHeight);

    else:

        numPoints = 8;

        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription")
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence")
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription")
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4")

        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]

        points = validate_clockwise_points(points)

        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(points[0],points[1],imWidth,imHeight);
            validate_point_inside_bounds(points[2],points[3],imWidth,imHeight);
            validate_point_inside_bounds(points[4],points[5],imWidth,imHeight);
            validate_point_inside_bounds(points[6],points[7],imWidth,imHeight);


    if withConfidence:
        try:
            confidence = float(m.group(numPoints+1))
        except ValueError:
            raise Exception("Confidence value must be a float")

    if withTranscription:
        posTranscription = numPoints + (2 if withConfidence else 1)
        transcription = m.group(posTranscription)
        m2 = re.match(r'^\s*\"(.*)\"\s*$',transcription)
        if m2 != None : #Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")

    return points,confidence,transcription


def validate_point_inside_bounds(x,y,imWidth,imHeight):
    if(x<0 or x>imWidth):
            raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" %(xmin,imWidth,imHeight))
    if(y<0 or y>imHeight):
            raise Exception("Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s" %(ymin,imWidth,imHeight))

def order_points(pts: np.ndarray):
    # # initialzie a list of coordinates that will be ordered
    # # such that the first entry in the list is the top-left,
    # # the second entry is the top-right, the third is the
    # # bottom-right, and the fourth is the bottom-left
    # rect = np.zeros((4, 2), dtype = "float32")

    # # the top-left point will have the smallest sum, whereas
    # # the bottom-right point will have the largest sum
    # s = pts.sum(axis = 1)
    # rect[0] = pts[np.argmin(s)]
    # rect[2] = pts[np.argmax(s)]

    # # now, compute the difference between the points, the
    # # top-right point will have the smallest difference,
    # # whereas the bottom-left will have the largest difference
    # diff = np.diff(pts, axis = 1)
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]

    # # return the ordered coordinates
    # return rect

    dirs = np.zeros_like(pts)
    mags = np.zeros((4,), dtype=pts.dtype)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]

        dir = p2 - p1
        mag = np.linalg.norm(dir)
        dir /= (mag + 1e-6)

        dirs[i] = dir
        mags[i] = mag

    # First, determine the ordering of the points. If clockwise (what we want),
    # then the cross products will all be positive, and if ccw, then negative.
    crosses = np.zeros((4,), dtype=pts.dtype)
    for i in range(4):
        d1 = dirs[i]
        d2 = dirs[(i + 1) % 4]

        cp = d1[0] * d2[1] - d1[1] * d2[0]

        crosses[i] = cp

    # print(f'dirs:\n{dirs}')
    # print(f'crosses:\n{crosses}')

    # if np.any(crosses == 0):
    #     raise ValueError("Invalid quad! - Simplified geo has less than 4 vertices.")

    # if not (np.all(crosses >= 0) or np.all(crosses <= 0)):
    #     raise ValueError("Invalid quad! - Inconsistent angle.")

    if np.sum(crosses) < 0:
        # Re-run this with the point order reversed
        return order_points(pts[::-1])

    # Ok, at this point, the points are defined in clockwise order, so the next
    # bit is a heuristic, that prefers p0->p1 to be the longest edge, while
    # also being towards the top-left of the image

    a_len = mags[0] + mags[2]
    b_len = mags[1] + mags[3]

    # print(f'A_len: {a_len}, B_len: {b_len}')

    if b_len > a_len * 2.0:
        # Rotate by one
        p2 = np.zeros_like(pts)
        for i in range(4):
            p2[(i + 1) % 4] = pts[i]
        return order_points(p2)

    # Finally, figure out which long-point is closer to top-left
    a_sum = pts[0,0] + pts[0,1]
    b_sum = pts[2,0] + pts[2,1]

    if a_sum < b_sum:
        return pts

    # Rotate by 2 positions
    p2 = np.zeros_like(pts)
    for i in range(4):
        p2[(i + 2) % 4] = pts[i]
    return p2

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
                [int(points[0]) , int(points[1])],
                [int(points[2]) , int(points[3])],
                [int(points[4]) , int(points[5])],
                [int(points[6]) , int(points[7])]
            ]
    # edge = [
    #             ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
    #             ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
    #             ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
    #             ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    # ]

    # summatory = edge[0] + edge[1] + edge[2] + edge[3];
    # if summatory>0:
    #     raise Exception("Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")

    point = order_points(np.array(point).astype(np.float32)).astype(np.int32)

    return point.reshape(-1)

def get_tl_line_values_from_file_contents(content,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True):
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    lines = content.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != "") :
            points, confidence, transcription = get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight);
            pointsList.append(points)
            transcriptionsList.append(transcription)
            confidencesList.append(confidence)

    if withConfidence and len(confidencesList)>0 and sort_by_confidences:
        import numpy as np
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    return pointsList,confidencesList,transcriptionsList

def main_evaluation(p,default_evaluation_params_fn,validate_data_fn,evaluate_method_fn,show_result=True,per_sample=True):
    """
    This process validates a method, evaluates it and if it succed generates a ZIP file with a JSON entry for each sample.
    Params:
    p: Dictionary of parmeters with the GT/submission locations. If None is passed, the parameters send by the system are used.
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    evaluate_method_fn: points to a function that evaluated the submission and return a Dictionary with the results
    """

    if (p == None):
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        if(len(sys.argv)<3):
            print_help()

    evalParams = default_evaluation_params_fn()
    if 'p' in p.keys():
        evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]) )

    resDict={'calculated':True,'Message':'','method':'{}','per_sample':'{}'}
    try:
        validate_data_fn(p['g'], p['s'], evalParams)
        evalData = evaluate_method_fn(p['g'], p['s'], evalParams)
        resDict.update(evalData)

    except Exception as e:
        resDict['Message']= str(e)
        resDict['calculated']=False

    if 'o' in p:
        if not os.path.exists(p['o']):
            os.makedirs(p['o'])

        resultsOutputname = p['o'] + '/results.zip'
        outZip = zipfile.ZipFile(resultsOutputname, mode='w', allowZip64=True)

        del resDict['per_sample']
        if 'output_items' in resDict.keys():
            del resDict['output_items']

        outZip.writestr('method.json',json.dumps(resDict))

    if not resDict['calculated']:
        if show_result:
            sys.stderr.write('Error!\n'+ resDict['Message']+'\n\n')
        if 'o' in p:
            outZip.close()
        return resDict

    if 'o' in p:
        if per_sample == True:
            for k,v in evalData['per_sample'].items():
                outZip.writestr( k + '.json',json.dumps(v))

            if 'output_items' in evalData.keys():
                for k, v in evalData['output_items'].items():
                    outZip.writestr( k,v)

        outZip.close()

    if show_result:
        sys.stdout.write("Calculated!")
        sys.stdout.write(json.dumps(resDict['method']))

    return resDict


def main_validation(default_evaluation_params_fn,validate_data_fn):
    """
    This process validates a method
    Params:
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    """
    try:
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        evalParams = default_evaluation_params_fn()
        if 'p' in p.keys():
            evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]) )

        validate_data_fn(p['g'], p['s'], evalParams)
        print('SUCCESS')
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(101)
