from hachoir.parser import createParser
from hachoir.metadata import extractMetadata


# Get duration video file
def get_duration_ms(filename):
    h=0
    min=0
    sec=0
    ms=0

    filename, realname = filename, filename
    parser = createParser(filename, realname)

    metadata = extractMetadata(parser)
    text = metadata.exportPlaintext()
    duration = text[1].split(' ')
    for i in range(len(duration)):
        if duration[i] =='hours':
            h=3600000*int(duration[i-1])
        elif duration[i] =='min':
            min=60000*int(duration[i-1])
        elif duration[i] =='sec':
            sec=1000*int(duration[i-1])
        elif duration[i] =='ms':
            ms=int(duration[i-1])

    return h+min+sec+ms






#pathname ='/home/kseniia/Desktop/test_new_work/people-counting-opencv/videos/test.mp4' #Duration: 1 hours 57 min 45 sec     24 min 28 sec 592 ms
#meta = get_duration_ms(pathname)
#print(meta)