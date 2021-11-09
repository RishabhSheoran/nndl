import cv2
import numpy as np
import youtube_dl

if __name__ == '__main__':

    with open('links.txt') as file:
        urls = file.readlines()
    
    for index, video_url in enumerate(urls):
        
        code = video_url[video_url.index('=')+1:]
    
        ydl_opts = {}
        i = 0 # frame count
        ydl = youtube_dl.YoutubeDL(ydl_opts)
        info_dict = ydl.extract_info(video_url, download=False)

        formats = info_dict.get('formats',None)

        for f in formats:

            if f.get('format_note',None) == '144p':
                url = f.get('url',None)
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print('Video not opened')
                    exit(-1)
                count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    # cv2.imshow('frame', frame)
                    count += 1
                    if count % 60 == 0:
                        file_name = 'data/{}_{}_{}.jpg'.format(index, code, str(i))
                        cv2.imwrite(file_name, frame)
                        i += 1
                        count = 0
                    if cv2.waitKey(30)&0xFF == ord('q'): break

                cap.release()

        cv2.destroyAllWindows()