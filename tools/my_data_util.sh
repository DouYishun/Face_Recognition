#! /bin/bash
i=1
while(($i<=40))
do
    ./hibImport.sh -f /home/douyishun/Documents/data/face_data/orl_faces/s$i /user/douyishun/data/face/orl/s$i.hib
    let i++
done
