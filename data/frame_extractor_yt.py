import cv2
import numpy as np
import youtube_dl

if __name__ == '__main__':

    video_url = 'https://www.youtube.com/watch?v=8wewPn7TZfs'
    # video_url = 'https://www.youtube.com/watch?v=Xtwar56r4Lg'
    # video_url = 'https://www.youtube.com/watch?v=c3hh1AYMo7o'

    ydl_opts = {}
    i = 214  # frame count
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(video_url, download=False)

    formats = info_dict.get('formats', None)

    for f in formats:

        if f.get('format_note', None) == '144p':
            url = f.get('url', None)
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print('Video not opened')
                exit(-1)
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # cv2.imshow('frame', frame)
                count += 1
                if count % 60 == 0:
                    cv2.imwrite('data/0-8wewPn7TZfs-frame'+str(i)+'.jpg', frame)
                    i += 1
                    count = 0
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            # release VideoCapture
            cap.release()

    cv2.destroyAllWindows()
