MESH=1000
XCEN=0
YCEN=0
FRAMES=2000
RANGE=200
INFILE="100m_in_2000_1d"
OUTFILE="100m_in_2000_1d"
NOI=2000
TITLE="100m_in_2000_1d"
FINALNAME="100m_in_2000_1d"
TITLE1="100m"
TITLE2="in"
TITLE3="2000-Fragments(1Day)"
SPEEDOFSOUND=0.340
SLEEPTIME=0
FPS=100

# alias activate="C:\Users\kavis\Python_venv\ucsb\Scripts\activate"
source "C:/Users/kavis/Python_venv/ucsb/Scripts/activate"
echo $INFILE $MESH $FRAMES $RANGE $TITLE $OUTFILE $XCEN $YCEN $FPS $NOI $TITLE1 $TITLE2 $TITLE3 | python3 lightpulse-f.py

