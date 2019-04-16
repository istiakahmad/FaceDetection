import face_recognition
import cv2

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
i_image = face_recognition.load_image_file("i1.jpg")
i_face_encoding = face_recognition.face_encodings(i_image)[0]

print(type(i_face_encoding))
print(i_face_encoding)

# Create arrays of known face encodings and their names
known_face_encodings = [
    i_face_encoding
]

known_face_names = [
    "Ahmad"
]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

count= 0 # Frame Count
falsecount = 0
flag = 0 # Identification Value


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #..

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) #..

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        flag = 0
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            if count % 10 == 0:
                cv2.imwrite("data/" + str(count) + ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 10])
            count += 1

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                flag=1
            face_names.append(name)
            print("Face Identified Value: " + str(flag))

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        cv2.rectangle(frame, (left, bottom - 6), (right, bottom), (0, 0, 255), cv2.FILLED)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(falsecount)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

