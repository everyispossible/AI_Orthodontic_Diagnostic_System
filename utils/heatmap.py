import numpy as np
def fill_gaussian(heatmap, center,sigma):
    x0,y0=center
    s=sigma
    r=int(3*s)
    xx,yy=np.ogrid[-r:r+1,-r:r+1]
    gaussian=np.exp(-(xx*xx+yy*yy)/(2*s**2))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    height, width = heatmap.shape
    left,right=min(x0,r),min(width-x0,r+1)
    top,bottom=min(y0,r),min(height-y0,r+1)

    if left+right>0 and top+bottom>0:
        masked_heatmap=heatmap[y0-top:y0+bottom,x0-left:x0+right]
        masked_gaussian=gaussian[r-top:r+bottom,r-left:r+right]
        np.maximum(masked_heatmap,masked_gaussian,out=masked_heatmap)

def generate_heatmap(landmarks,h,w):
    num_points = landmarks.shape[0]
    heatmaps = np.zeros((num_points, h, w),dtype=np.float32)
    for i,(x,y) in enumerate(landmarks):
        center=(int(x+0.5),int(y+0.5))

        if 0<center[0]<w and 0<=center[1]<h:
            fill_gaussian(heatmaps[i],center,sigma=2.0)
    return  heatmaps