#! /bin/zsh
# Files
directory=/Volumes/Data/RGB/CrA/mosaic/
name_j=${directory}CrA_RGB_J
name_h=${directory}CrA_RGB_H
name_ks=${directory}CrA_RGB_Ks
name_l=${directory}CrA_RGB_L
names_all=("$name_j" "$name_h" "$name_ks" "$name_l")
#numbers=(00 01 02 10 11 12 20 21 22)
numbers=(11 01 02 10 00 12 20 21 22)
path_config=/Volumes/Data/RGB/CrA/mosaic/stiff_manual.config

# Set extraction levels (from auto mode on big tif file -3)
#l_min=-0.6
l_min=-3.6
l_max=1650

#r_min=0.5
r_min=-2.5
r_max=1750

#g_min=-1.4
g_min=-4.4
g_max=1725

#b_min=-0.8
b_min=-3.8
b_max=1400

# Loop over files
for num in "${numbers[@]}" ; do
  for name in "${names_all[@]}" ; do

    tifname=${name}_${num}.tif
    fitsname=${name}_${num}.fits

    if [[ $name == *"_Ks"* ]] ; then
      stiff -c $path_config "$fitsname" -OUTFILE_NAME "$tifname" -MIN_LEVEL $r_min -MAX_LEVEL $r_max
    elif [[ $name == *"_H"* ]] ; then
      stiff -c $path_config "$fitsname" -OUTFILE_NAME "$tifname" -MIN_LEVEL $g_min -MAX_LEVEL $g_max
    elif [[ $name == *"_J"* ]] ; then
      stiff -c $path_config "$fitsname" -OUTFILE_NAME "$tifname" -MIN_LEVEL $b_min -MAX_LEVEL $b_max
    elif [[ $name == *"_L"* ]] ; then
      stiff -c $path_config "$fitsname" -OUTFILE_NAME "$tifname" -MIN_LEVEL $l_min -MAX_LEVEL $l_max
    fi
  done
done
