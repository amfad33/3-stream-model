import pandas as pd
from pytube import YouTube
import os.path


# Youtube downloader for HVU and Youtube 8M. Need to fix paths/part numbers accordingly
def title_gen(title, id):
    path = 'G:\\HVU Downloader\\train\\part ' + str(id) + '\\'
    title = title.replace("#", "")
    title = title.replace("<", "")
    title = title.replace("$", "")
    title = title.replace("+", "")
    title = title.replace("%", "")
    title = title.replace(">", "")
    title = title.replace("!", "")
    title = title.replace("`", "")
    title = title.replace("&", "")
    title = title.replace("*", "")
    title = title.replace("\'", "")
    title = title.replace("|", "")
    title = title.replace("{", "")
    title = title.replace("?", "")
    title = title.replace("\"", "")
    title = title.replace("=", "")
    title = title.replace("}", "")
    title = title.replace("/", "")
    title = title.replace(":", "")
    title = title.replace("\\", "")
    title = title.replace("@", "")
    while "  " in title:
        title = title.replace("  ", " ")
    if os.path.isfile(path + title + ".mp4"):
        id = 1
        while os.path.isfile(path + title + "_" + str(id) + ".mp4"):
            id += 1
            continue
        return title + "_" + str(id)
    else:
        return title


j = 1  # start part
while j < 483:   # end part   for hvu

    path = 'G:\\HVU Downloader\\train\\part ' + str(j)

    path_final = r'G:\HVU Downloader\HVU_Train_part' + str(j) + '.csv'   # parted csv
    try:
        part = pd.read_csv(path_final, encoding='utf-8', index_col='id')   # if parted csv exists load to continue
        part = part[part['youtube_id'].str.len() == 11]

    except Exception as e:   # if parted csv dont exist create it
        print(e)
        df = pd.read_csv(r'G:\HVU Downloader\HVU_Train_V1.0.csv', encoding='utf-8')
        part = df[((j-1)*1000):(j*1000)]   # take 1000 videos
        # part = df[70986:71000]
        part = part.dropna()
        part = part[part['youtube_id'].str.len() == 11]
        part["title"] = ""
        part["file"] = False

    text = ""

    i = (j-1)*1000
    # i = 70986

    for x in ("https://www.youtube.com/watch?v=" + part['youtube_id']):

        if part.at[i, 'file']:
            i += 1
            if i % 10 == 0:
                print(str((i - ((j - 1) * 1000)) / 10) + "% done")
            continue

        loop = True
        while (loop):
            loop = False
            print(x)
            text += x + "\n"
            try:
                # object creation using YouTube
                # which was imported in the beginning
                yt = YouTube(x, use_oauth=True, allow_oauth_cache=True)
            except:
                # to handle exception
                print("Connection Error")
                continue
                # filters out all the files with "mp4" extension

            # get the video with the extension and
            # resolution passed in the get() function
            # d_video = yt.streams.get(mp4files[-1].extension, mp4files[-1].resolution)
            try:
                # downloading the video
                mp4files = yt.streams.filter(progressive=True, file_extension="mp4", resolution="480p").first()

                if mp4files.filesize > 5e+7:
                    # print("480p File size big!")
                    try:
                        # downloading the video
                        mp4files = yt.streams.filter(progressive=True, file_extension="mp4", resolution="360p").first()

                        if mp4files.filesize <= 5e+7:
                            # d_video = mp4files.get_by_resolution("480p")
                            title = title_gen(mp4files.title, j)
                            mp4files.download(output_path=path, filename=title + ".mp4")
                            print("File id " + part['index'] + " with name " + title + " downloaded.")
                            part.at[i, 'title'] = title
                            part.at[i, 'file'] = True
                    except Exception as e:
                        print(e)
                        if "[WinError 10054]" in str(e):
                            loop = True
                            continue

                        # print("Some Error when downloading 360p!")
                        part.at[i, 'title'] = ""
                        part.at[i, 'file'] = False
                else:
                    title = title_gen(mp4files.title, j)
                    mp4files.download(output_path=path, filename=title + ".mp4")
                    print("File " + title + " downloaded.")
                    part.at[i, 'title'] = title
                    part.at[i, 'file'] = True
            except Exception as e:
                print(e)
                if "[WinError 10054]" in str(e):
                    loop = True
                    continue
                # print("Some Error when downloading 480p trying 360p!")
                try:
                    # downloading the video
                    mp4files = yt.streams.filter(progressive=True, file_extension="mp4", resolution="360p").first()
                    if mp4files.filesize <= 5e+7:
                        # d_video = mp4files.get_by_resolution("480p")
                        title = title_gen(mp4files.title, j)
                        mp4files.download(output_path=path, filename=title + ".mp4")
                        print("File " + title + " downloaded.")
                        part.at[i, 'title'] = title
                        part.at[i, 'file'] = True
                except Exception as e:
                    print(e)
                    if "[WinError 10054]" in str(e):
                        loop = True
                        continue
                    # print("Some Error when downloading 360p!")
                    part.at[i, 'title'] = ""
                    part.at[i, 'file'] = False
        part.to_csv(r'G:\HVU Downloader\HVU_Train_part' + str(j) + '.csv', encoding='utf-8')
        i += 1
        if i % 10 == 0:
            print(str((i-((j-1)*1000))/10) + "% done")
    # print(text)
    # part.to_csv(r'E:\HVU Downloader\HVU_Train_test.csv', encoding='utf-8')
    j += 1
