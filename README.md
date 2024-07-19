# Automatic-BPMN
Generate BPMN using the recording of meeting notes

**Install required libraries**

pip install git+https://github.com/openai/whisper.git -q
pip install nlp-id
pip install Sastrawi
pip install reportlab
pip install BeautifulSoup4
pip install ttkthemes

**Usage**
1. Open Terminal:
Use the Anaconda prompt for best compatibility.

2. Run the Application:
Copy code **"python app.py"**

3. Input the Recording:
When prompted, choose the recording file (in .mp4 format) and choose start the process button.

4. Process the Recording:
The application will process the recording and generate an XML file containing the BPMN diagram.

5. View the Result:
Open the generated XML file using the Bizagi Modeler BPMN tool to manage and refine the BPMN view.
