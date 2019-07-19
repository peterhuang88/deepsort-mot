import sys
import os
import glob
from urllib.request import urlopen
import CAM2CameraDatabaseAPIClient as cam2
from CAM2ImageArchiver.CAM2ImageArchiver import CAM2ImageArchiver


# Admin creds
clientID = 'cdd56c76857988df2683998b14fc29dbef0727d87ba89878d613bc8437f77aa761f104f8d19c2d5044824eb527539b22'
clientSecret = '066fb15b7ef6966ffaee61f7e83a5d6f4bd1bf4117db961d6d2816679ef01786255ff9d'
db = cam2.Client(clientID, clientSecret)
cams = db.search_camera(city='NY')
# url = 'http://' + cam['reference_url']

# Start queuing all the cameras in the database
# for cam_dict in db_data:
    # print(cam_dict['retrieval'], '\n')

id = '5b0cfa8045bb0c0004277e37'
for i in range(15):
    cam = db.camera_by_id(id)
    camList = [cam, ]
    # check if cam is active (urllib) before archiving
    archiver = CAM2ImageArchiver(num_processes=1)
    archiver.archive(camList, duration=1, interval=1)

    renameFiles = glob.glob('results/'+id+'/'+id+'*')
    if len(renameFiles) != 0:
        os.rename(renameFiles[0], 'results/'+id+'/'+ str(i) +'.png')
exit()


# listHTML = []
# i = 0
# for cam in cams:
#     print(i)
#     i+=1
#     url = 'https://' + cam['reference_url']
#     try:
#         html = urlopen(url, timeout=5)
#         listHTML.append(url)
#     except:
#         print('passed')
#         pass
#
# pp(listHTML)
