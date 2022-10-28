from numpy import absolute
from vb2py.vbfunctions import Double,vbObjectInitialize,Long,vbForRange,Variant,Abs
import math


def pchipend(h1, h2, del1, del2):
    D = Double()
    D = ( ( 2 * h1 + h2 )  * del1 - h1 * del2 )  /  ( h1 + h2 )
    if ( D * del1 < 0 ) :
        D = 0
    elif  ( ( del1 * del2 < 0 )  and  ( Abs(D) > Abs(3 * del1) ) ) :
        D = 3 * del1
    fn_return_value = D
    return fn_return_value

def bdrint(rate, dist, low, high):
    log_rate = vbObjectInitialize(((1, 4),), Double)
    log_dist = vbObjectInitialize(((1, 4),), Double)
    i = Long()
    H = vbObjectInitialize(((1, 3),), Double)
    delta = vbObjectInitialize(((1, 3),), Double)
    D = vbObjectInitialize(((1, 4),), Double)
    C = vbObjectInitialize(((1, 3),), Double)
    B = vbObjectInitialize(((1, 3),), Double)
    s0 = Double()
    s1 = Double()

    result = Double()
    for i in vbForRange(1, 4):
        log_rate[i] = math.log10(rate[5 - i])
        log_dist[i] = dist[5 - i]
    for i in vbForRange(1, 3):
        H[i] = log_dist(i + 1) - log_dist(i)
        delta[i] = ( log_rate(i + 1) - log_rate(i) )  / H(i)
    D[1] = pchipend(H(1), H(2), delta(1), delta(2))
    for i in vbForRange(2, 3):
        D[i] = ( 3 * H(i - 1) + 3 * H(i) )  /  ( ( 2 * H(i) + H(i - 1) )  / delta(i - 1) +  ( H(i) + 2 * H(i - 1) )  / delta(i) )
    D[4] = pchipend(H(3), H(2), delta(3), delta(2))
    for i in vbForRange(1, 3):
        C[i] = ( 3 * delta(i) - 2 * D(i) - D(i + 1) )  / H(i)
        B[i] = ( D(i) - 2 * delta(i) + D(i + 1) )  /  ( H(i) * H(i) )
    # cubic function is rate(i) + s*(d(i) + s*(c(i) + s*(b(i))) where s = x - dist(i)
    # or rate(i) + s*d(i) + s*s*c(i) + s*s*s*b(i)
    # primitive is s*rate(i) + s*s*d(i)/2 + s*s*s*c(i)/3 + s*s*s*s*b(i)/4
    result = 0
    for i in vbForRange(1, 3):
        s0 = log_dist(i)
        s1 = log_dist(i + 1)
        # clip s0 to valid range
        s0 = max(s0, low)
        s0 = min(s0, high)
        # clip s1 to valid range
        s1 = max(s1, low)
        s1 = min(s1, high)
        s0 = s0 - log_dist(i)
        s1 = s1 - log_dist(i)
        if ( s1 > s0 ) :
            result = result +  ( s1 - s0 )  * log_rate(i)
            result = result +  ( s1 * s1 - s0 * s0 )  * D(i) / 2
            result = result +  ( s1 * s1 * s1 - s0 * s0 * s0 )  * C(i) / 3
            result = result +  ( s1 * s1 * s1 * s1 - s0 * s0 * s0 * s0 )  * B(i) / 4
    fn_return_value = result
    return fn_return_value

def bdrate(rateA, distA, rateB, distB):
    minPSNR = Double()
    maxPSNR = Double()
    vA = Double()
    vB = Double()
    avg = Double()
    minPSNR = max(min(distA), min(distB))
    maxPSNR = min(max(distA), max(distB))
    vA = bdrint(rateA, distA, minPSNR, maxPSNR)
    vB = bdrint(rateB, distB, minPSNR, maxPSNR)
    avg = ( vB - vA )  /  ( maxPSNR - minPSNR )
    fn_return_value = 10**avg - 1
    return fn_return_value


def bubbleSortTwo(A, B, elements):
    n = len(A)
 
    # Traverse through all array elements
    for i in range(n-1):
    # range(n) also work but outer loop will
    # repeat one time more than needed.
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if A[j] > A[j + 1] :
                A[j], A[j + 1] = A[j + 1], A[j]
                B[j], B[j + 1] = B[j + 1], B[j]

def addValues(rate, dist, log_rate, log_dist, elements):
    i = Long()
    j = Long()
    j = elements
    i = 1
    for j,_ in enumerate(rate):
        # Add elements only if they are not empty
        if not (rate[j]==0 and dist[j]==0):
            log_rate[i] = math.log10(rate[j])
            log_dist[i] = dist[j]
            i = i + 1
            j = j - 1
    elements = i - 1

# VB2PY (UntranslatedCode) Argument Passing Semantics / Decorators not supported: log_rate - ByRef 
# VB2PY (UntranslatedCode) Argument Passing Semantics / Decorators not supported: log_dist - ByRef 
# VB2PY (UntranslatedCode) Argument Passing Semantics / Decorators not supported: elements - ByRef 
def removeDuplicates(log_rate, log_dist, elements):
    dElements = Long()
    cElements = Long()
    dElements = 1
    cElements = elements - 1
    i = 1
    while i < cElements + 1:
        if ( log_rate(i) == log_rate(i + 1) and log_dist(i) == log_dist(i + 1) ):
            del log_rate[i+1]
            del log_dist[i+1]
            cElements = cElements - 1
            dElements = dElements + 1
        else:
            dElements = dElements + 1
        i = i + 1
    elements = dElements

def intCurve(xArr, yArr, low, high, elements):
    H = vbObjectInitialize(objtype=Double)
    delta = vbObjectInitialize(objtype=Double)
    D = vbObjectInitialize(objtype=Double)
    C = vbObjectInitialize(objtype=Double)
    B = vbObjectInitialize(objtype=Double)
    s0 = Double()
    s1 = Double()
    result = Double()
    #MsgBox elements
    H = vbObjectInitialize((elements + 2,), Variant)
    delta = vbObjectInitialize((elements + 2,), Variant)

    for i in vbForRange(1, elements - 1):
        #For i = LBound(xArr) To UBound(xArr)
        H[i] = xArr(i + 1) - xArr(i)
        delta[i] = ( yArr(i + 1) - yArr(i) )  / H(i)
    #MsgBox elements
    D = vbObjectInitialize((elements + 2,), Variant)
    D[1] = pchipend(H(1), H(2), delta(1), delta(2))
    for i in vbForRange(2, elements - 1):
        #For i = LBound(xArr) + 1 To UBound(xArr)
        D[i] = ( 3 * H(i - 1) + 3 * H(i) )  /  ( ( 2 * H(i) + H(i - 1) )  / delta(i - 1) +  ( H(i) + 2 * H(i - 1) )  / delta(i) )
    D[elements] = pchipend(H(elements - 1), H(elements - 2), delta(elements - 1), delta(elements - 2))
    C = vbObjectInitialize((elements + 2,), Variant)
    B = vbObjectInitialize((elements + 2,), Variant)
    for i in vbForRange(1, elements - 1):
        #For i = LBound(xArr) To UBound(xArr)
        C[i] = ( 3 * delta(i) - 2 * D(i) - D(i + 1) )  / H(i)
        B[i] = ( D(i) - 2 * delta(i) + D(i + 1) )  /  ( H(i) * H(i) )
    # cubic function is rate(i) + s*(d(i) + s*(c(i) + s*(b(i))) where s = x - dist(i)
    # or rate(i) + s*d(i) + s*s*c(i) + s*s*s*b(i)
    # primitive is s*rate(i) + s*s*d(i)/2 + s*s*s*c(i)/3 + s*s*s*s*b(i)/4
    result = 0
    # Compute rate for the extrapolated region if needed
    s0 = xArr(1)
    s1 = xArr(2)
    for i in vbForRange(1, elements - 1):
        #For i = LBound(xArr) To UBound(xArr)
        s0 = xArr(i)
        s1 = xArr(i + 1)
        # clip s0 to valid range
        s0 = max(s0, low)
        s0 = min(s0, high)
        # clip s1 to valid range
        s1 = max(s1, low)
        s1 = min(s1, high)
        s0 = s0 - xArr(i)
        s1 = s1 - xArr(i)
        if ( s1 > s0 ) :
            result = result +  ( s1 - s0 )  * yArr(i)
            result = result +  ( s1 * s1 - s0 * s0 )  * D(i) / 2
            result = result +  ( s1 * s1 * s1 - s0 * s0 * s0 )  * C(i) / 3
            result = result +  ( s1 * s1 * s1 * s1 - s0 * s0 * s0 * s0 )  * B(i) / 4
    fn_return_value = result
    return fn_return_value

def bdRIntEnh(rate, dist, low, high):
    elements = Long()
    log_rate = vbObjectInitialize(objtype=Double)
    log_dist = vbObjectInitialize(objtype=Double)
    elements = len(rate)
    log_rate = vbObjectInitialize((elements,), Variant)
    log_dist = vbObjectInitialize((elements,), Variant)
    addValues(rate=rate, dist=dist, log_rate=log_rate, log_dist=log_dist, elements=elements)
    # Sort the data in case they were not placed in order
    bubbleSortTwo(A=log_dist, B=log_rate, elements=elements)

    # Remove duplicates
    removeDuplicates(log_rate=log_rate, log_dist=log_dist, elements=elements)
    # If plots do not overlap, extend range
    # Extrapolate towards the minimum if needed
    if ( log_dist(1) > low ) :
        for i in vbForRange(1, elements):
            log_rate[elements + 2 - i] = log_rate(elements + 1 - i)
            log_dist[elements + 2 - i] = log_dist(elements + 1 - i)
        elements = elements + 1
        log_dist[1] = low
        log_rate[1] = log_rate(2) +  ( low - log_dist(2) )  *  ( log_rate(2) - log_rate(3) )  /  ( log_dist(2) - log_dist(3) )
    # Extrapolate towards the maximum if needed
    if ( log_dist[elements] < high ) :
        log_dist[elements + 1] = high

        log_rate[elements + 1] = log_rate(elements) +  ( high - log_dist(elements) )  *  ( log_rate(elements) - log_rate(elements - 1) )  /  ( log_dist(elements) - log_dist(elements - 1) )
        elements = elements + 1
    fn_return_value = intCurve(log_dist, log_rate, low, high, elements)
    return fn_return_value



def bdRateExtend(rateA, distA, rateB, distB, bMode='None', bRange=False):
    minPSNRA = Double()
    maxPSNRA = Double()
    minPSNRB = Double()
    maxPSNRB = Double()
    minMinPSNR = Double()
    maxMinPSNR = Double()
    minMaxPSNR = Double()
    maxMaxPSNR = Double()
    minPSNR = Double()
    maxPSNR = Double()
    minPSNRA = min(distA)
    maxPSNRA = max(distA)
    minPSNRB = min(distB)
    maxPSNRB = max(distB)
    minMinPSNR = min(minPSNRA, minPSNRB)
    maxMinPSNR = max(minPSNRA, minPSNRB)
    minMaxPSNR = min(maxPSNRA, maxPSNRB)
    maxMaxPSNR = max(maxPSNRA, maxPSNRB)
    minPSNR = minMinPSNR
    maxPSNR = minMaxPSNR
    if ( bRange == True ) :
        if ( minPSNRA > maxPSNRB or minPSNRB > maxPSNRA ) :
            # No overlap case
            fn_return_value = 0
        else:
            # Give the overlap percentage
            fn_return_value = ( maxPSNR - maxMinPSNR )  /  ( maxPSNRA - minPSNRA )
            if ( maxPSNRB < maxPSNRA ) :
                fn_return_value = - 1 * bdRateExtend()
        return fn_return_value
    else:
        # Here is we handle all possible modes for
        # the computation. By default no extrapolation
        # is performed. The "always" methods force extrapolation
        # while the Low/High/Both methods do so adaptively.
        if ( bMode == 'LowAlways' ) :
            minPSNR = minMinPSNR
            maxPSNR = minMaxPSNR
        elif  ( bMode == 'HighAlways' ) :
            minPSNR = maxMinPSNR
            maxPSNR = maxMaxPSNR
        elif  ( bMode == 'BothAlways' ) :
            minPSNR = minMinPSNR
            maxPSNR = maxMaxPSNR
        elif  ( bMode == 'None' or not  ( minPSNRA > maxPSNRB or minPSNRB > maxPSNRA ) ) :
            if ( minPSNRA > maxPSNRB ) :
                fn_return_value = 1
                return fn_return_value
            elif  ( minPSNRB > maxPSNRA ) :
                fn_return_value = - 1
                return fn_return_value
            else:
                minPSNR = maxMinPSNR
                maxPSNR = minMaxPSNR
        elif  ( bMode == 'Low' ) :
            minPSNR = minMinPSNR
            maxPSNR = minMaxPSNR
        elif  ( bMode == 'High' ) :
            minPSNR = maxMinPSNR
            maxPSNR = maxMaxPSNR
        elif  ( bMode == 'Both' ) :
            minPSNR = minMinPSNR
            maxPSNR = maxMaxPSNR
        vA = bdRIntEnh(rateA, distA, minPSNR, maxPSNR)
        vB = bdRIntEnh(rateB, distB, minPSNR, maxPSNR)
        avg = ( vB - vA )  /  ( maxPSNR - minPSNR )
        if avg<=10:
            fn_return_value = (10**avg) - 1
        else:
            fn_return_value = 1
    return fn_return_value

def calculateAnchorMetric(metric_file, image ,quality_mapped):
    bdrate = []
    with open(metric_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "\t" in line.strip():
                l = line.strip().split("\t")
            else:
                l = line.strip().split(" ")
            l = list(filter(None,l))
            if l[0] == image:
                l[1:] = list(map(float, l[1:]))
                bdrate.append(l)
    anchorMetric = bdrate[0 if 5-quality_mapped == -1 else 5-quality_mapped]
    anchorMetric = {'image':anchorMetric[0],'q':anchorMetric[1],'bpp':anchorMetric[2],'y_msssim':anchorMetric[3],\
                    'psnr':{'y':anchorMetric[4],'u':anchorMetric[5],'v':anchorMetric[6]},\
                    'vif':anchorMetric[7],'fsim':anchorMetric[8],'nlpd':anchorMetric[9],\
                    'iw_ssim':anchorMetric[10],'vmaf':anchorMetric[11],'psnr_vhsm':anchorMetric[12],\
                    'enc':anchorMetric[13],'dec':anchorMetric[14],'dec_mem':anchorMetric[15]
                }
    return anchorMetric
                
def calAverageBdRate (metric_file, metric2compare,quality_mapped, skip = [], MonotonicCheck = True, includeChroma = False):
    r'''format of the metric file must be as below, imagename TAB sth TAB bitrate TAB msssim TAB Y-PSNR ...
        the last tree columns are not important. you can copy from excel to create this file.
    00001_TE_2096x1400.png	11	0.82699	0.99678963	33.51856382	40.65607509	40.93702368	0.60751206	0.99924552	0.08423211	0.99586618	92.77	35.22424023	1.79	0.7	0.52
    00001_TE_2096x1400.png	8	0.51124	0.99450159	31.05198105	39.15312151	39.38953222	0.52930456	0.99826419	0.11956127	0.99108827	89.058616	31.62693589	1.88	0.63	0.52
    00001_TE_2096x1400.png	4	0.24507	0.9828707	26.93977924	37.6031455	37.63766744	0.39632055	0.99473655	0.19779179	0.97150558	75.544426	25.62292495	1.96	0.61	0.51
    00001_TE_2096x1400.png	2	0.12208	0.95525426	24.13536484	35.27702298	35.773628	0.29006395	0.98659801	0.28616774	0.92809504	56.90251	21.1103723	1.4	0.57	0.52
    00001_TE_2096x1400.png	1	0.06037	0.91522169	22.46460085	34.99313235	35.40764734	0.20030689	0.96842182	0.37917146	0.85250336	34.800955	18.43379971	1.37	0.54	0.51

    metric2compare is a dictionary with the following entries:
        metric2compare['image'],metric2compare['q'],
        metric2compare['bpp'],metric2compare['y_msssim'],metric2compare['psnr']['y'],metric2compare['psnr']['u'],
        metric2compare['psnr']['v'],metric2compare['vif'],metric2compare['fsim'],metric2compare['nlpd'],
        metric2compare['iw_ssim'],metric2compare['vmaf'],metric2compare['psnr_vhsm'],
    
    quality_mapped is 1,2,3,4 or 6.
    ...
    '''
    
    if metric2compare['y_msssim'] < 0.4 or metric2compare['vmaf']<10 or metric2compare['psnr_vhsm']<10: #problem in encoding/decoding
        return 100
    
    bdrate = []
    with open(metric_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "\t" in line.strip():
                l = line.strip().split("\t")
            else:
                l = line.strip().split(" ")
            l = list(filter(None,l))
            if l[0] == metric2compare['image']:
                l[1:] = list(map(float, l[1:]))
                bdrate.append(l)
    
    toCompare = [
                    metric2compare['image'],metric2compare['q'],\
                    metric2compare['bpp'],metric2compare['y_msssim'],metric2compare['psnr']['y'],metric2compare['psnr']['u'],\
                    metric2compare['psnr']['v'],metric2compare['vif'],metric2compare['fsim'],metric2compare['nlpd'],\
                    metric2compare['iw_ssim'],metric2compare['vmaf'],metric2compare['psnr_vhsm'],\
                    0,0,0,
                ]
    for ii in range(3,len(toCompare)):
        toCompare[ii] = float(int(toCompare[ii]*100000000))/100000000
    toCompare[2] = float(int(toCompare[2]*100000))/100000

    bdrate_compare = bdrate.copy()
    bdrate_compare[0 if 5-quality_mapped == -1 else 5-quality_mapped] = toCompare.copy()
    
    bdrate_compare = list(zip(*bdrate_compare))
    bdrate_t = list(zip(*bdrate))
    summ = 0
    devide = 0
    for j in range(3,len(bdrate_compare)-3):
        if 'y_msssim' in skip and j == 3:
            continue
        if j == 4 or j==5 or j==6:
            continue
        if 'vif' in skip and j == 7:
            continue
        if 'fsim' in skip and j == 8:
            continue
        if 'nlpd' in skip and j == 9:
            continue
        if 'iw_ssim' in skip and j == 10:
            continue   
        if 'vmaf' in skip and j == 11:
            continue
        if 'psnr_hvsm' in skip and j == 12:
            continue   
        gain = bdRateExtend(bdrate_t[2],bdrate_t[j],bdrate_compare[2],bdrate_compare[j])
        sanityCheck = True
        if j != 9:
            for kk in range(4):
                if bdrate_compare[j][kk] < bdrate_compare[j][kk+1]:
                    sanityCheck = not MonotonicCheck
        if sanityCheck:
            devide += 1
            summ += gain
    if devide > 0:
        summ = summ/devide
    else:
        return 100
    if includeChroma:
        secondary = ((sum(bdrate_compare[5]) + sum(bdrate_compare[6])) - (sum(bdrate_t[5]) + sum(bdrate_t[6])))/(sum(bdrate_t[5]) + sum(bdrate_t[6]))
        secondary = -secondary/3
        summ += secondary
    return summ


    

def calAverageBdRate2 (metric_file, metricList, skip = [], MonotonicCheck = True, includeChroma = False):
    bdrate = []
    with open(metric_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "\t" in line.strip():
                l = line.strip().split("\t")
            else:
                l = line.strip().split(" ")
            l = list(filter(None,l))
            if l[0] == metricList[0]['image']:
                l[1:] = list(map(float, l[1:]))
                bdrate.append(l)
    bdrate_compare = []
    for metric2compare in metricList:
        toCompare = [
                        metric2compare['image'],metric2compare['q'],\
                        metric2compare['bpp'],metric2compare['y_msssim'],metric2compare['psnr']['y'],metric2compare['psnr']['u'],\
                        metric2compare['psnr']['v'],metric2compare['vif'],metric2compare['fsim'],metric2compare['nlpd'],\
                        metric2compare['iw_ssim'],metric2compare['vmaf'],metric2compare['psnr_vhsm'],\
                        metric2compare['enc'],metric2compare['dec'],metric2compare['dec_mem'],
                    ]
        bdrate_compare.append(toCompare)
    
    bdrate_compare_t = list(zip(*bdrate_compare))
    bdrate_t = list(zip(*bdrate))
    summ = 0
    devide = 0
    for j in range(3,len(bdrate_compare_t)-3):
        if 'y_msssim' in skip and j == 3:
            continue
        if j == 4 or j == 5 or j == 6:
            continue
        if 'vif' in skip and j == 7:
            continue
        if 'fsim' in skip and j == 8:
            continue
        if 'nlpd' in skip and j == 9:
            continue
        if 'iw_ssim' in skip and j == 10:
            continue   
        if 'vmaf' in skip and j == 11:
            continue
        if 'psnr_hvsm' in skip and j == 12:
            continue   
        
        gain = bdRateExtend(bdrate_t[2],bdrate_t[j],bdrate_compare_t[2],bdrate_compare_t[j])
        sanityCheck = True
        if j != 9:
            for kk in range(4):
                if bdrate_compare_t[j][kk] < bdrate_compare_t[j][kk+1]:
                    sanityCheck = not MonotonicCheck
        if sanityCheck:
            devide += 1
            summ += gain
    if devide > 0:
        summ = summ/devide
    else:
        return 100
    if includeChroma:
        secondary = ((sum(bdrate_compare_t[5]) + sum(bdrate_compare_t[6])) - (sum(bdrate_t[5]) + sum(bdrate_t[6])))/(sum(bdrate_t[5]) + sum(bdrate_t[6]))
        secondary = -secondary/3
        summ += secondary
    return summ, bdrate_compare