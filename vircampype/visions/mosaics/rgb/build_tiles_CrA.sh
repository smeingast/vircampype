# Files
directory=/Volumes/Data/RGB/CrA/mosaic/
name_j=${directory}CrA_RGB_J
name_h=${directory}CrA_RGB_H
name_ks=${directory}CrA_RGB_Ks
name_l=${directory}CrA_RGB_L
names_all=("$name_j" "$name_h" "$name_ks" "$name_l")

# Define x cuts
x0=1 ; x1=18001 ; x2=42001 ; naxis1=60628
dx0=$((x1-x0-1))
dx1=$((x2-x1-1))
dx2=$((naxis1-x2))

# Define y cuts
y0=1 ; y1=30001 ; y2=60001 ; naxis2=79393
dy0=$((y1-y0-1))
dy1=$((y2-y1-1))
dy2=$((naxis2-y2))

# Loop over files
for name in "${names_all[@]}" ; do

    # Current filename with extension
    in=${name}.fits

    # Create tiles
    nn=00
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x0 $y0 $dx0 $dy0
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=10
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x1 $y0 $dx1 $dy0
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=20
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x2 $y0 $dx2 $dy0
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=01
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x0 $y1 $dx0 $dy1
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=11
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x1 $y1 $dx1 $dy1
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=21
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x2 $y1 $dx2 $dy1
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=02
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x0 $y2 $dx0 $dy2
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=12
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x1 $y2 $dx1 $dy2
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

    nn=22
    mSubImage -p "$in" "${name}"_"${nn}"_64.fits $x2 $y2 $dx2 $dy2
    mConvert -b -32 "${name}"_"${nn}"_64.fits "${name}"_"${nn}".fits
    rm "${name}"_"${nn}"_64.fits

#    mSubImage -p $in ${name}_10.fits $x1 $y0 $dx1 $dy0
#    mSubImage -p $in ${name}_20.fits $x2 $y0 $dx2 $dy0
#    mSubImage -p $in ${name}_01.fits $x0 $y1 $dx0 $dy1
#    mSubImage -p $in ${name}_11.fits $x1 $y1 $dx1 $dy1
#    mSubImage -p $in ${name}_21.fits $x2 $y1 $dx2 $dy1
#    mSubImage -p $in ${name}_02.fits $x0 $y2 $dx0 $dy2
#    mSubImage -p $in ${name}_12.fits $x1 $y2 $dx1 $dy2
#    mSubImage -p $in ${name}_22.fits $x2 $y2 $dx2 $dy2

done
