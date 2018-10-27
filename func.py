
y=0
z=0
l=0
r=0
t=0
b=0
def detect_bot(x):
    for box, classx in x.items() :
        if 'bottle' in classx[0]:
            global z
            global y
            global l
            global r
            global t
            global b
            y1, x1, y2, x2 = box
            (l, r, t, b) = (x1 * 640, x2 * 640, y1 * 480, y2 * 480)
            
            y=(l+r)/2
            
            z=(t+b)/2
    
      
        