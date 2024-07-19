import sys
import tkinter as tk
from tkinter import filedialog, ttk
from ttkthemes import ThemedTk
import threading
import subprocess
import os
import whisper
import copy
from bs4 import BeautifulSoup


from collections import OrderedDict
from nlp_id.postag import PosTag
postagger = PosTag()
import nltk
nltk.download('punkt')

import re
import pandas as pd
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

from datetime import datetime
import pytz


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
model = whisper.load_model('small')


def import_file(label_status, selected_file):
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        # Mengonversi path absolut ke path relatif terhadap direktori skrip
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.relpath(file_path, script_dir)
        print('relative path')
        print(relative_path)

        selected_file[0] = relative_path
        label_status.config(text=f"File dipilih: {selected_file[0]}")



def start_processing(label_status, selected_file, progressbar):
    if not selected_file[0]:
        label_status.config(text="Pilih file MP4 terlebih dahulu!")
        return
    
    label_status.config(text="Proses sedang berjalan...")
    progressbar.start()

    # Menjalankan fungsi pemrosesan di latar belakang
    thread = threading.Thread(target=run_processing, args=(label_status, selected_file, progressbar))
    thread.start()



def run_processing(label_status, selected_file, progressbar):
    create_meeting_notes(selected_file[0])
    
    # Update UI setelah proses selesai
    progressbar.stop()
    label_status.config(text="Proses Selesai!")

def convert_path(path_file, output_ext="mp4"):
  filename, ext = os.path.splitext(path_file)
  subprocess.call(["ffmpeg", "-y", "-i", path_file, f"{filename}.{output_ext}"],
                  stdout=subprocess.DEVNULL,
                  stderr=subprocess.STDOUT,
                  shell=True)
  return f"{filename}.{output_ext}"


def generate_meeting_notes(output_name, data):
  discussion_index = data.lower().find("hasil diskusi")

  meeting_note = {}
  if discussion_index != -1:
    pre_discussion = data[:discussion_index].strip()
    discussion_section = data[discussion_index:].strip()

    index_sehingga = list(re.finditer('sehingga', discussion_section))
    positions_sehingga = [match.start() for match in index_sehingga]

    for ind_shg in positions_sehingga:
      if(discussion_section[ind_shg-2] == '.'):
        discussion_section = discussion_section[:ind_shg-2] + ','+ discussion_section[ind_shg-1:]

    index_kompleksitas = list(re.finditer('kompleksitas', discussion_section))
    positions_kompleksitas = [match.start() for match in index_kompleksitas]

    for ind in positions_kompleksitas:
      if(discussion_section[ind-2] != '.'):
        discussion_section = discussion_section[:ind-1] + '.' + discussion_section[ind-1:]

    index_proses_dimulai = list(re.finditer('proses dimulai', discussion_section))
    positions_proses_dimulai = [match.start() for match in index_proses_dimulai]

    for ind_a in positions_proses_dimulai:
      if(discussion_section[ind_a-2] != '.'):
        discussion_section = discussion_section[:ind_a-1] + '.' + discussion_section[ind_a-1:]

    date, attendee, topic, iteration, activity, requirement = get_detail(pre_discussion)
    user_stories, priorities, complexities, story_descriptions = get_result_discussion(discussion_section)

    if(not user_stories):
      user_stories = ['']
    if(not priorities):
      priorities = ['']
    if(not complexities):
      complexities = ['']
    if(not story_descriptions):
      story_descriptions = ['']

    if(len(requirement) > 1):
      requirement = ','.join([item[0] for item in requirement])

    df = pd.DataFrame({
      'Tanggal': pd.Series(date, index=range(len(date))),
      'Peserta': pd.Series(attendee, index=range(len(attendee))),
      'Agenda': pd.Series(topic, index=range(len(topic))),
      'Iterasi': pd.Series(iteration, index=range(len(iteration))),
      'Kegiatan': pd.Series(activity, index=range(len(activity))),
      'Kebutuhan': pd.Series(requirement, index=range(1)),
      'Hasil Diskusi' : ''
    })
    if(len(attendee) > 1):
      df['Peserta'] = df['Peserta'].apply(lambda x: ', '.join(x))
    df = df.transpose()

    temp = create_variable(len(user_stories))


    for i in range(len(user_stories)):
      temp[i] = pd.DataFrame({
        'Id User Story': pd.Series(i+1, index=range(1)),
        'Prioritas': pd.Series(priorities[i], index=range(1)),
        'Kompleksitas': pd.Series(complexities[i], index=range(1)),
        'User Story': pd.Series(user_stories[i], index=range(1)),
        'Story Description': pd.Series(story_descriptions[i], index=range(1)),
      }).transpose()


    if(len(temp) > 1):
      df1 = pd.DataFrame()
      for i in range(1,len(temp),1):
        if(i > 1):
          df1 = pd.concat([df1,temp[i]],axis = 0)
        else:
          df1 = pd.concat([temp[i-1],temp[i]],axis = 0)
    else:
      df1 = temp[0]

    df = pd.concat([df,df1],axis = 0)
    df.to_csv('file.csv')
    generate_meeting_notes_pdf('file.csv', output_name)
    return story_descriptions


def wrap_text(text, width):
    # Memecah teks menjadi baris-baris yang lebih pendek
    wrapped_text = textwrap.wrap(text, width=width)
    # Menggabungkan baris-baris menjadi satu teks dengan pemisah newline
    wrapped_text = "\n".join(wrapped_text)
    return wrapped_text

def generate_meeting_notes_pdf(csv_file, output_name):
    # Baca data dari file CSV menggunakan pandas
    df = pd.read_csv(csv_file)

    # Memecah teks panjang dalam setiap kolom menjadi beberapa baris
    width = 80  # Lebar maksimum setiap baris
    for col in df.columns:
        df[col] = df[col].apply(lambda x: wrap_text(str(x), width) if pd.notnull(x) else "")

    # Ubah DataFrame menjadi list of lists untuk membuat tabel PDF
    data_table = [df.columns.tolist()] + df.values.tolist()

    # Buat dokumen PDF
    pdf = SimpleDocTemplate(output_name, pagesize=letter)

    # Buat tabel dari data
    table = Table(data_table)

    # Atur gaya untuk tabel
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    table.setStyle(style)

    # Tambahkan tabel ke konten PDF
    pdf.build([table])


def create_variable(len):
  variabel = []

  for i in range(len):
    variabel.append(f"temp_{i}")

  return variabel


def get_detail(pre_discussion):
  sentence_contains = []
  sentence_details = []
  date = []
  attendee = []
  topic = []
  iteration = []
  activity = []
  requirement = []

  array_of_sentence_prediscussion = text_to_array(pre_discussion)

  for index, sentence in enumerate(array_of_sentence_prediscussion):
    if('pada tanggal' in sentence):
      tanggal_index = sentence.find("pada tanggal")
      date.append(sentence[tanggal_index + len("pada tanggal"):].strip())
    elif any(keyword in sentence for keyword in ['peserta meeting', 'peserta rapat', 'peserta diskusi', 'paserta meeting']):
      attendee.append(get_data_with_sequence(sentence))
    elif('membahas mengenai' in sentence or 'membahas tentang' in sentence):
      topic.append(getTopic(sentence))
    elif('iterasi' in sentence or 'pertemuan' in sentence):
      iteration.append(sentence.split("pertemuan")[1] if "pertemuan" in sentence else sentence.split("iterasi")[1] if "iterasi" in sentence else "")
    elif('kegiatan' in sentence):
      activity_index = sentence.find("adalah")
      activity.append(sentence[activity_index + len("adalah"):].strip())
    elif('kebutuhan' in sentence):
      requirement.append(get_data_with_sequence(sentence))

  return date, attendee, topic, iteration, activity, requirement

def get_result_discussion(discussion_section):
  user_story_id = []
  user_stories = []
  updated_user_stories = []
  story_descriptions = []
  priorities = []
  complexities = []
  index_story_desc = []
  index_start_index_start_desc = 0
  index_end_desc = 0


  array_of_sentence_discussion_section = text_to_array(discussion_section)

  for index, sentence in enumerate(array_of_sentence_discussion_section):
    if('sebagai' in sentence and 'saya ingin' in sentence and 'sehingga' in sentence):
      user_stories.append({'sentence' : sentence, 'index' : index})
    elif('proses dimulai' in sentence):
      index_story_desc.append(index)
    elif('kompleksitas' in sentence):
      complexity_index = sentence.find("kompleksitas")
      complexities.append(sentence[complexity_index + len("kompleksitas"):].strip())
    elif('kompeksitas' in sentence):
      complexity_index = sentence.find("kompeksitas")
      complexities.append(sentence[complexity_index + len("kompeksitas"):].strip())
    elif('prioritas' in sentence):
      priority_index = sentence.find("prioritas")
      priorities.append(sentence[priority_index + len("prioritas"):].strip())


  pairs = []
  for i in range(len(index_story_desc)):
    if i + 1 < len(user_stories):
      pairs.append((index_story_desc[i], user_stories[i+1]['index']))
    else:
      pairs.append((index_story_desc[i], len(array_of_sentence_discussion_section)))


  for pair in pairs:
    story_descriptions.append(". ".join(array_of_sentence_discussion_section[pair[0]:pair[1]]))

  for user_story in user_stories:
    updated_user_stories.append(user_story['sentence'])

  return updated_user_stories, priorities, complexities, story_descriptions

def get_data_with_sequence(sentence):
  sequence = ["pertama", "kedua", "ketiga", "keempat", "kelima" ,"keenam", "ketujuh", "kedelapan", "kesembilan", "kesepuluh"]
  found = False

  for word in sequence:
    if word in sentence:
      found = True
      break

  if(not found):
    index = sentence.find("adalah")
    return sentence[index + len("adalah"):].strip()
  else:
    hasil = []
    for i in range(len(sequence) - 1):
      index1 = sentence.find(sequence[i])
      index2 = sentence.find(sequence[i+1])
      if index1 != -1 and index2 != -1:
          index1 += len(sequence[i])  # Menggeser indeks ke setelah kata kunci
          # Mengambil kata di antara kata kunci
          kata_di_antara = sentence[index1:index2].strip()
          hasil.append(kata_di_antara)
      elif index1 != -1 and index2 == -1:
        kata_di_antara = sentence[index1 + len(sequence[i]):].strip()
        hasil.append(kata_di_antara)

    # Menampilkan hasil
    result = []
    for i, kata in enumerate(hasil, 1):
      result.append(f"{i}. {kata}")

    return result

def getTopic(sentence):
  topic = ''
  topic_index = sentence.find("membahas mengenai")
  if topic_index == -1:
    topic_index = sentence.find("membahas tentang")
    topic = sentence[topic_index + len("membahas tentang"):].strip()
  else:
    topic = sentence[topic_index + len("membahas mengenai"):].strip()
  return topic

fw_words = ['login','tiba']

def change_general_fw_word_tag(pos_tag):
  for tag in pos_tag:
    for data in tag:
      if data['word'] in fw_words:
        data['tag'] = 'VB'

def text_to_array(text):
  array_of_sentence = text.split('.')
  for i in range(len(array_of_sentence)):
    array_of_sentence[i] = array_of_sentence[i].strip()
    if(array_of_sentence[i] == ''):
      array_of_sentence.pop(i)
  return array_of_sentence


def pos_tag(array_of_sentence):
  pos_tagging_list = []
  for sentence in array_of_sentence:
    pos_tagging_list.append(postagger.get_pos_tag(sentence))

  pos_tagging_dicts = []
  for sentence in pos_tagging_list:
    sentence_dict_list = []
    for word, tag in sentence:
        sentence_dict_list.append({'word': word, 'tag': tag})
    pos_tagging_dicts.append(sentence_dict_list)
  return pos_tagging_dicts

def pos_tag_sentence(sentence):
  pos_tag_list = []
  pos_tagging = postagger.get_pos_tag(sentence)
  for word, tag in pos_tagging:
    pos_tag_list.append({'word': word, 'tag': tag})
  return pos_tag_list

def convert_to_detail(array_of_sentence, pos_tagging):
  detailed_array = []
  sentence_structure = {'subject': '', 'verb': '', 'object': ''}

  for i in range(len(array_of_sentence)):
    sentence = array_of_sentence[i]
    pos_tag = pos_tagging[i]
    detailed_array.append({'sentence': sentence, 'tags': pos_tag, 'structure': sentence_structure})
  return detailed_array

def convert_passive_verb_to_active(word):
  kata_dasar = stemmer.stem(word)
  result_kata_dasar = word

  postfix = word.split(kata_dasar)[1]

  # meng- apabila bentuk dasar berawal dengan fonem: /vokal/, /k/ disertai hilangnya /k/, /g/, /kh/ , /h/.
  if kata_dasar[0] in ['a', 'i', 'u', 'e', 'o','g','h']:  # Periksa apakah dimulai dengan fonem vokal
      result_kata_dasar = 'meng' + kata_dasar
  elif kata_dasar.startswith('k'):  # Periksa apakah dimulai dengan "k"
      if(kata_dasar.startswith(('kr', 'kl'))):
        result_kata_dasar = 'meng' + kata_dasar
      else:
        result_kata_dasar = 'meng' + kata_dasar[1:]

  # mem- apabila bentuk dasar berawal dengan fonem: /p/ disertai hilangnya /p/, /b/, /f/.
  if kata_dasar[0] in ['b','f']:  # Periksa apakah dimulai dengan fonem vokal
      result_kata_dasar = 'mem' + kata_dasar
  elif kata_dasar.startswith('p'):
      if kata_dasar.startswith('pr'):
        result_kata_dasar = 'mem' + kata_dasar
      else:
        result_kata_dasar = 'mem' + kata_dasar[1:]

  # men- apabila bentuk dasar berawal dengan fonem: /t/ disertai hilangnya /t/, /d/.
  if kata_dasar[0] in ['d','c']:  # Periksa apakah dimulai dengan fonem vokal
      result_kata_dasar = 'men' + kata_dasar
  elif kata_dasar.startswith('t'):
      if kata_dasar.startswith('tr'):
        result_kata_dasar = 'men' + kata_dasar
      else:
        result_kata_dasar = 'men' + kata_dasar[1:]

  # meny- apabila bentuk dasar berawal dengan fonem: /s/ disertai hilangnya /s/, /c/, /j/, /h/.
  if kata_dasar.startswith('s'):
      if kata_dasar.startswith(('st', 'sk', 'sp')):
        result_kata_dasar = 'men' + kata_dasar
      else:
        result_kata_dasar = 'meny' + kata_dasar[1:]

  # me- apabila bentuk dasar berawal dengan fonem: /l/, /r/, /w/, /n/.
  if kata_dasar[0] in ['l','r','w','n']:  # Periksa apakah dimulai dengan fonem vokal
      result_kata_dasar = 'me' + kata_dasar

  # print(result_kata_dasar)
  return result_kata_dasar + postfix

def find_blank_structure(index_current_sentence,detailed_array):
  subject = detailed_array[index_current_sentence]['structure']['subject']
  verb = detailed_array[index_current_sentence]['structure']['verb']
  obyek = detailed_array[index_current_sentence]['structure']['object']

  if(index_current_sentence + 1 < len(detailed_array) and detailed_array[index_current_sentence+1]['tags'][-1]['word'] == 'changed'):
    if(detailed_array[index_current_sentence]['structure']['subject'] != ''):
      subject = detailed_array[index_current_sentence]['structure']['subject']
    elif(detailed_array[index_current_sentence+1]['structure']['subject'] != ''):
      subject = detailed_array[index_current_sentence+1]['structure']['subject']
    else:
      subject = detailed_array[index_current_sentence-1]['structure']['subject']
  if(index_current_sentence + 1 < len(detailed_array) and obyek is None and subject.lower() != 'proses'):
    obyek = detailed_array[index_current_sentence+1]['structure']['object']

  detailed_array[index_current_sentence]['structure'] = {'subject': subject, 'verb': verb, 'object': obyek}


def find_structure(current_sentence_tag, index_current_sentence, detailed_array):

  verb = None
  subject = None
  obyek = None
  index_of_nn = None
  index_of_vb = None
  type_of_sentence = ''

  for i in range(len(current_sentence_tag)):
    if(current_sentence_tag[i]['tag'].lower() == 'vb'):
      index_of_vb = i
      verb = current_sentence_tag[i]['word']
      break

  if(index_of_vb is not None):
    if(current_sentence_tag[index_of_vb]['word'].startswith('di')):
      for i in range(index_of_vb, len(current_sentence_tag), 1):
        if(current_sentence_tag[i]['tag'].lower() == 'nn'):
          # Subyek sclalu tcrdiri atas kata benda atau kata ganti
          index_of_nn = i
          subject = current_sentence_tag[i]['word']
          type_of_sentence = 'passive'

          verb = convert_passive_verb_to_active(verb)
          # print('verb baru: ', verb)
          break
    else:
      for i in range(index_of_vb):
        if(current_sentence_tag[i]['tag'].lower() == 'nn'):
          # Subyek sclalu tcrdiri atas kata benda atau kata ganti
          index_of_nn = i
          subject = current_sentence_tag[i]['word']
          type_of_sentence = 'active'
          break

    if(subject is None):
      if(current_sentence_tag[index_of_vb]['word'].startswith('di')):
        type_of_sentence = 'passive'
        verb = convert_passive_verb_to_active(verb)
      else:
        type_of_sentence = 'active'
      if(index_current_sentence + 1 < len(detailed_array) and detailed_array[index_current_sentence+1]['tags'][-1]['word'] == 'changed'):
        subject = ''
      else:
        subject = detailed_array[index_current_sentence-1]['structure']['subject']

    # find object
    if(type_of_sentence == 'active'):
      for i in range(index_of_vb, len(current_sentence_tag), 1):
        if(current_sentence_tag[i]['tag'].lower() == 'nn'):
          if(i+1 < len(current_sentence_tag) and current_sentence_tag[i+1]['tag'].lower() == 'in'):
            obyek = current_sentence_tag[i]['word'] + ' ' + current_sentence_tag[i+1]['word'] + ' ' + current_sentence_tag[i+2]['word']
          else:
            obyek = current_sentence_tag[i]['word']
          break
    else:
      for i in range(index_of_vb):
        if(current_sentence_tag[i]['tag'].lower() == 'nn'):
          obyek = current_sentence_tag[i]['word']
          break

    detailed_array[index_current_sentence]['structure'] = {'subject': subject, 'verb': verb, 'object': obyek}

  return {'subject': subject , 'verb': verb, 'object':obyek}

def handle_clause_sentence(current_sentence_tag, index_current_sentence):
  index_of_akibat = None
  sc_found_sebab = False

  for index, word in enumerate(current_sentence_tag):
    if(word['word'] in  ['jika','seandainya','kalau']):
      sc_found_sebab = True

  if(sc_found_sebab):
    for index, word in enumerate(current_sentence_tag):
      if(word['word'] == 'maka'):
        index_of_akibat = index

  title = ' '.join(current_sentence_tag[i]['word'].replace('_', ' ') for i in range (1,index_of_akibat,1))

  new_sentence = [];
  if(index_of_akibat is not None):
    for i in range(index_of_akibat + 1, len(current_sentence_tag), 1):
      new_sentence.append(current_sentence_tag[i])

    # result = findSubjectForClauseSentence(index_of_akibat, new_sentence, current_sentence_tag, index_current_sentence)
    sentence = ' '.join(current_sentence_tag[i]['word'].replace('_', ' ') for i in range (index_of_akibat,len(current_sentence_tag),1))


    return {'data': new_sentence, 'type': 'clauseSentence', 'case' : 'jika ' + title, 'sentence' : sentence}
  return

def split_sentence_cc(index_of_sign_compund_sentence, current_sentence_tag, index, clause_sentence):
  newSentence = []
  subject = ''
  vb_candidate_new_sentence = False
  nn_candidate_new_sentence = False
  placing_index = 0

  # if(clause_sentence):
  #   handle_clause_sentence(current_sentence_tag, index)
  for i in range(index_of_sign_compund_sentence + 1, len(current_sentence_tag), 1): #+1 agar kata 'cc' tidak include
    if current_sentence_tag[i]['tag'].lower() == 'vb':
      vb_candidate_new_sentence = True
    elif current_sentence_tag[i]['tag'].lower() == 'nn':
      nn_candidate_new_sentence = True

  if(vb_candidate_new_sentence and nn_candidate_new_sentence):
    for i in range(index_of_sign_compund_sentence +  1, len(current_sentence_tag), 1): # +1 agar kata 'cc' tidak include
      newSentence.append(current_sentence_tag[i])
    for i in range(len(newSentence), 0, -1):
      current_sentence_tag.pop(-i)

    # print(current_sentence_tag)
    if(current_sentence_tag[index_of_sign_compund_sentence]['word'] == 'setelah'):
      placing_index = index
    else:
      placing_index = index + 1


    for i in range(len(current_sentence_tag)):
      if(i != 0):
        if current_sentence_tag[i]['word'] == 'setelah':
          current_sentence_tag[i]['word'] = 'changed'

    return {'data': newSentence, 'type': 'newSentence', 'index': placing_index}

    # result = findSubjectForCompoundSentences(index_of_sign_compund_sentence, newSentence, current_sentence_tag, index, None)
    # print(result)
    # if(result is not None):
    #   if(result['type'] == 'use previous subject'):
    #     newSentence.insert(0, result['subject'])
    # for i in range(len(newSentence), 0, -1):
    #   current_sentence_tag.pop(-i)
    # return {'data': newSentence, 'type': 'newSentence', 'index': index+1, 'subject': result['subject']['word'],  'verb': result['verb']['word']}
  else: # setelah kata 'dan' bukan kalimat majemuk
      current_sentence_tag[index_of_sign_compund_sentence]['word'] = current_sentence_tag[index_of_sign_compund_sentence - 1]['word'] + ' ' + current_sentence_tag[index_of_sign_compund_sentence]['word'] + ' ' + current_sentence_tag[index_of_sign_compund_sentence+1]['word']
      current_sentence_tag[index_of_sign_compund_sentence]['tag'] = 'NN'
      new_datas = [current_sentence_tag[i] for i in range(len(current_sentence_tag)) if i not in [index_of_sign_compund_sentence - 1, index_of_sign_compund_sentence + 1]]
      return {'data' : new_datas, 'type': 'updateData'}
  
def get_sentence_before_in_yang(data):
  tag_object = pos_tag_sentence(data['structure']['object'])

  word_before_in_tag = []
  for index, word_tag in enumerate(tag_object):
    if word_tag['tag'] == 'IN':
      word_before_in_tag = [tag_object[k]['word'] for k in range(index)]
      break
    else:
      word_before_in_tag = [word_tag['word'] for word_tag in tag_object]

  obyek = ' '.join(word_before_in_tag)

  bagian_kalimat = obyek.split(' yang')
  result = bagian_kalimat[0].strip()

  return result

def remove_same_process(detailed_array):
  temp = copy.deepcopy(detailed_array)
  index_removed = []
  for i in range(0,len(temp)):
    temp[i]['structure']['verb'] = stemmer.stem(temp[i]['structure']['verb'])
    if(temp[i]['structure']['object'] is not None):
      temp[i]['structure']['object'] = get_sentence_before_in_yang(temp[i])

  for i in range(1,len(temp)):
    if(temp[i]['tags'][0]['word'] == 'setelah'):
      first_structure = temp[i-1]['structure']
      second_structure = temp[i]['structure']
      if(first_structure == second_structure):
        index_removed.append(i)

  return set(index_removed)



def find_index(param, detailed_array):
  index_found = []
  for index, data in enumerate(detailed_array):
    if(data.get(param)):
      index_found.append(index)

  return index_found


def grouping_index(grouped_process_titik_temu, search_process):
  # 1. Hitung jumlah grup dalam grouped_cases
  num_groups = len(grouped_process_titik_temu)

  # 2. Bagi index_cases menjadi grup
  grouped = []
  start_index = 0
  for group in grouped_process_titik_temu:
      num_elements = len(group)
      grouped.append(search_process[start_index:start_index+num_elements])
      start_index += num_elements
  return grouped

def assign_parent_case(index_cases, detailed_array):
  array_stem_verb = copy.deepcopy(detailed_array)
  index_of_first_case = index_cases[0]
  index_of_second_case = index_cases[1]

  temp = detailed_array[index_of_second_case-1]
  temp['structure']['verb'] =  stemmer.stem(temp['structure']['verb'])

  for i in range(len(array_stem_verb)):
    array_stem_verb[i]['structure']['verb'] = stemmer.stem(array_stem_verb[i]['structure']['verb'])

  for i in range(index_of_second_case, len(detailed_array), 1):
    if(temp['structure'] == array_stem_verb[i]['structure'] and (detailed_array[i].get('sign') != 'end')):
      detailed_array[i]['titik_temu_proses'] = detailed_array[index_of_first_case]['case'] + ' dan ' + detailed_array[index_of_second_case]['case']
      detailed_array[index_of_second_case-1]['titik_temu_proses'] = detailed_array[index_of_first_case]['case'] + ' dan ' + detailed_array[index_of_second_case]['case']

def remove_duplicate_item(array):
  unique_data = []
  indexes_seen = set()
  for item in array:
    index = item['index']
    if index not in indexes_seen:
        unique_data.append(item)
        indexes_seen.add(index)


  return unique_data

def find_back_process(current_sentence_tag):
  for data in current_sentence_tag:
    if('kembali' in data['word'] ):
      return True
  return False

def compound_word_rule(prev_word, curr_word, mid_word=None):
    valid_combinations = [('nn', 'nn'), ('nn', 'jj'), ('jj', 'nn'), ('vb', 'vb')]

    if mid_word is not None:
      if(mid_word['word'] == 'yang'):
        valid_combinations.extend([('nn', 'yang', 'jj'), ('nn', 'yang', 'vb')])
        return (prev_word['tag'].lower(), mid_word['word'].lower(), curr_word['tag'].lower()) in valid_combinations
    else:
      return (prev_word['tag'].lower(), curr_word['tag'].lower()) in valid_combinations


def compound(current_sentence_tag):
  compounds = []
  temp_curr_sentence = current_sentence_tag.copy()
  pos_compound = []



  for i in range(1, len(current_sentence_tag)):
    if current_sentence_tag[i-1]['word'] == 'yang':
      checkCompoundRule = compound_word_rule(current_sentence_tag[i-2], current_sentence_tag[i], current_sentence_tag[i-1])
      if checkCompoundRule:
        temp_curr_sentence[i-2]['word'] = current_sentence_tag[i-2]['word'] + ' ' + current_sentence_tag[i-1]['word'] + ' ' + current_sentence_tag[i]['word']
        pos_compound.append(i-1)
        pos_compound.append(i)
    else:
      checkCompoundRule = compound_word_rule(current_sentence_tag[i-1], current_sentence_tag[i])
      if checkCompoundRule:
        temp_curr_sentence[i-1]['word'] = current_sentence_tag[i-1]['word'] + ' ' + current_sentence_tag[i]['word']
        pos_compound.append(i)

  new_datas = [current_sentence_tag[i] for i in range(len(current_sentence_tag)) if i not in pos_compound]
  return new_datas


def check_clause_sentence(current_sentence_tag, index):
  consist_sign_clause_sentence = False
  index_of_sign_clause_sentence = None

  for i in range(len(current_sentence_tag)):
    if((current_sentence_tag[i]['tag'].lower() == 'sc') and current_sentence_tag[i]['word'] in ['jika','seandainya','kalau']):
      consist_sign_clause_sentence = True
      index_of_sign_clause_sentence = i

  if(consist_sign_clause_sentence):
    return handle_clause_sentence(current_sentence_tag, index)
  return


def check_compound_sentence(current_sentence_tag, index):
  consist_sign_compund_sentence_tag = False
  index_of_sign_compund_sentence = None
  list_index_of_sign_compund_sentence = None
  clause_sentence = False

  for i in range(len(current_sentence_tag)):
    if((current_sentence_tag[i]['tag'].lower() == 'adv' and current_sentence_tag[i]['word'] in ['selanjutnya', 'kemudian', 'lalu']) or (current_sentence_tag[i]['tag'].lower() == 'sc' and current_sentence_tag[i]['word'] in ['sebelum', 'setelah']) or current_sentence_tag[i]['word'] == 'maka'):
      if(i > 1):
        index_of_sign_compund_sentence = i
        consist_sign_compund_sentence_tag = True
        break
    elif(current_sentence_tag[i]['tag'].lower() == 'cc' and  (current_sentence_tag[i]['word'].lower() == 'dan' or current_sentence_tag[i]['word'].lower() == 'serta')):
      index_of_sign_compund_sentence = i
      consist_sign_compund_sentence_tag = True
      break


  if(consist_sign_compund_sentence_tag):
    return split_sentence_cc(index_of_sign_compund_sentence, current_sentence_tag, index, clause_sentence)
  return


def remove_unused_process(detailed_array, temp_group_to_closed_gateway, index_back_process):
  index_to_remove = []
  for index, data in enumerate(temp_group_to_closed_gateway):
    for i in range(len(data)):
      if(i != len(data) - 1):
        index_to_remove.append(data[i])

  for index, data in enumerate(detailed_array):
    if(data.get('sign')):
      index_to_remove.append(index)

  for data in index_back_process:
    index_to_remove.append(data)


  return index_to_remove

def generate_xml(data):
  array_of_sentence = text_to_array(data)
  pos_tagging = pos_tag(array_of_sentence)
  change_general_fw_word_tag(pos_tagging)
  detailed_array = convert_to_detail(array_of_sentence, pos_tagging)
  print("hasil pos tagging :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")


  for data in detailed_array:
    updateData = compound(data['tags'])
    if(updateData):
      data['tags'] = updateData.copy()
  print("hasil compound :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")



  for index, data in enumerate(detailed_array):
    result = check_clause_sentence(data['tags'], index)
    if(result):
      detailed_array[index] = {'sentence' : result['sentence'],'tags' : result['data'],'structure' : {'subject' : '', 'verb' : '', 'object' : ''}}
      detailed_array[index]['case'] =  result['case']
  print("hasil penyesuaian (jika,maka) :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")


  # check kalimat majemuk, jika ada buat array baru
  for index, data in enumerate(detailed_array):
    result = check_compound_sentence(data['tags'], index)
    # print(result)
    if(result is not None):
      if(result['type'] == 'updateData'):
        data['tags'] = result['data'].copy()
      elif(result['type'] == 'newSentence'):
        if('case' in data and data['tags'][-1]['word'] == 'changed'):
          detailed_array.insert(result['index'], {'sentence' : ' '.join(item['word'].replace('_', ' ') for item in result['data']), 'tags' : result['data'] , 'case' : data['case'] ,'structure' : {'subject': '', 'verb': '', 'object': ''}})
          del data['case']
        else:
          detailed_array.insert(result['index'], {'sentence' : ' '.join(item['word'].replace('_', ' ') for item in result['data']), 'tags' : result['data'] , 'structure' : {'subject': '', 'verb': '', 'object': ''}})

  print("hasil penyederhanaan kalimat :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")


  for index, data in enumerate(detailed_array):
    words = [tag['word'] for tag in data['tags']]
    combined_tag = ' '.join(words)
    if('proses' in combined_tag and ('berakhir' in combined_tag or 'selesai' in combined_tag)):
      detailed_array[index]['sign'] =  'end'
    # tags = data['tags']
    # for i in range(len(tags) - 1):
    #   if 'proses' in tags[i]['word'] and ('berakhir' in tags[i+1]['word'] or 'selesai' in tags[i+1]['word'] or (i+2 < len(tags) and ('berakhir' in tags[i+2]['word'] or 'selesai' in tags[i+2]['word']))):
    #     detailed_array[index]['sign'] =  'end'



  for index, data in enumerate(detailed_array):
    temp = find_structure(data['tags'], index, detailed_array)


  for index, data in enumerate(detailed_array):
    find_blank_structure(index, detailed_array)

  index_removed = remove_same_process(detailed_array)
  detailed_array = [element for index, element in enumerate(detailed_array) if index not in index_removed]

  print("hasil pencarian struktur sintaksis kalimat :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")


  index_cases = find_index('case', detailed_array)

  # titik temu proses
  array = []
  for i in range(1, len(index_cases)):
    temp = detailed_array[index_cases[i]-1]['structure']
    if('proses' in temp['subject']): #jika temp adalah proses berakhir
      temp = detailed_array[index_cases[i]-2]['structure']
    for j in range(index_cases[i], len(detailed_array)):
        if(temp == detailed_array[j]['structure']):
          array.append({'structure': detailed_array[j]['structure'], 'index': j})
          array.append({'structure': detailed_array[index_cases[i]-1]['structure'], 'index': index_cases[i]-1})

  array = remove_duplicate_item(array)
  # Buat kamus kosong untuk menyimpan struktur unik
  unique_structures = {}

  # Iterasi melalui daftar abc
  for item in array:
    # Dapatkan kunci unik dari struktur
    key = (item['structure']['subject'], item['structure']['verb'], item['structure']['object'])

    # Periksa apakah kunci sudah ada dalam kamus
    if key in unique_structures:
        # Jika sudah ada, tambahkan indeks ke daftar indeks yang ada
        unique_structures[key]['index'].append(item['index'])
    else:
        # Jika belum ada, tambahkan struktur baru ke kamus dengan indeks awal
        unique_structures[key] = {'structure': item['structure'], 'index': [item['index']]}

  res = [{'structure': value['structure'], 'index': value['index']} for value in unique_structures.values()]

  if(res):
    for i, data in enumerate(res):
      for index in data['index']:
        detailed_array[index]['titik_temu_proses'] = 'titik temu jika maka ke '+str(i)

  print("hasil proses mencari titik temu proses :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")

  grouped_process_titik_temu = []
  for item in res:
    sorted_index = sorted(item['index'])
    grouped_process_titik_temu.append(sorted_index)
  grouped_process_to_cases = grouping_index(grouped_process_titik_temu, index_cases)

  # memastikan bahwa semua kasus terkelompok, dan jika ada kasus yang tidak termasuk dalam grup mana pun, mereka ditambahkan sebagai grup tersendiri
  temp = []
  temp.extend(grouped_process_to_cases)
  remaining_cases = set(index_cases)
  for group in grouped_process_to_cases:
      for case in group:
          if case in remaining_cases:
              remaining_cases.remove(case)
  if remaining_cases:
      temp.append(list(remaining_cases))

  # if(len(grouped_process_to_cases) == 0):
  #   if(index_cases):
  #     grouped_process_to_cases.append(index_cases)
  # grouped_process_to_cases


  # create open gateway
  for i in range(len(temp)):
    temp[i][0] = temp[i][0] + i # add i to find fit pos for gateway
    detailed_array.insert(temp[i][0], {'gateway': 'open gateway', 'cases':'', 'structure' : {'subject' : detailed_array[temp[i][0]-1]['structure']['subject']}})

  print("hasil penambahan array open gateway :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")

  index_titik_temu_proses = []
  for index, data in enumerate(detailed_array):
    if(data.get('titik_temu_proses')):
      index_titik_temu_proses.append(index)




  grouped_index_titik_temu_proses = grouping_index(grouped_process_titik_temu, index_titik_temu_proses)
  grouped_index_titik_temu_proses

  last_indexes = [group[-1] for group in grouped_index_titik_temu_proses]
  last_indexes

  for i, index in enumerate(last_indexes):
    index += i
    detailed_array.insert(index, {'gateway': 'closed gateway', 'case':'', 'structure' : {'subject' : detailed_array[index-1]['structure']['subject']}})

  print("hasil penambahan array closed gateway :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")

  index_back_process = []
  for index, data in enumerate(detailed_array):
    temp_det_arr = data.copy()
    if('tags' in data):
      result = find_back_process(data['tags'])
      if(result):
        index_back_process.append(index)
        if 'kembali' in data['structure']['verb']:
          data['structure']['verb'] = data['structure']['verb'].replace('kembali','')

        for i in range(len(detailed_array)):
          if('verb' in detailed_array[i]['structure']):
            temp_detailed_array = detailed_array[i].copy()
            temp_1 = stemmer.stem(detailed_array[i]['structure']['verb'])
            temp_2 = stemmer.stem(data['structure']['verb'])
            if(temp_detailed_array['structure']['object'] is not None):
              temp_detailed_array['structure']['object'] = get_sentence_before_in_yang(temp_detailed_array)
            if(temp_det_arr['structure']['object'] is not None):
              temp_det_arr['structure']['object'] = get_sentence_before_in_yang(temp_det_arr)

            if(temp_1 == temp_2 and temp_detailed_array['structure']['object'] == temp_det_arr['structure']['object']):

              if(detailed_array[index-1].get('sign') == 'end'):
                detailed_array[index-2]['back_to'] = i
                detailed_array[index-2]['from_index'] = index
              else:
                detailed_array[index-1]['back_to'] = i
                detailed_array[index-1]['from_index'] = index
              if('case' in data):
                detailed_array[i]['referenced_case'] = data['case']
              break


  for index, data in enumerate(detailed_array):
    data['index'] = index

  for item in detailed_array:
      # Hapus key 'tags' jika ada
      if 'tags' in item:
          del item['tags']


  index_cases_final = find_index('case', detailed_array)
  for data in detailed_array:
    for i, data_index in enumerate(index_cases_final):
      if('from_index' in data):
        if(data['from_index'] == data_index):
          index_cases_final[i] = data['back_to']

  group_after_open_gateway = grouping_index(temp, index_cases_final)

  # if(len(group_after_open_gateway) == 0):
  #   if(index_cases_final):
  #     group_after_open_gateway.append(index_cases_final)



  process_to_closed_gateway = []
  index_titik_temu_proses = find_index('titik_temu_proses', detailed_array)

  group_to_closed_gateway = grouping_index(grouped_process_titik_temu, index_titik_temu_proses)
  temp_group_to_closed_gateway = copy.deepcopy(group_to_closed_gateway)
  temp_2_group_to_closed_gateway = copy.deepcopy(group_to_closed_gateway)
  # print(group_to_closed_gateway)
  for i in range(len(group_to_closed_gateway)):
    # -1 karena polanya adalah pada if sebelum akhir, lanjut terus prosesnya sampai titik temu
    # -2 karena polanya adalah pada if terakhir, akan terbentuk gateway close sebelum titik temu -> jika -1 arahnya kegateway bukan proses
    for index, data_j in enumerate(group_to_closed_gateway[i]):
      if(index == len(group_to_closed_gateway[i])-1):
        temp_2_group_to_closed_gateway[i][index] = group_to_closed_gateway[i][index] -2
        if(detailed_array[group_to_closed_gateway[i][index] -2].get('titik_temu_proses')):
          group_to_closed_gateway[i][index] = group_after_open_gateway[i][0]-1 # jika saat di minus 2 adalah titik temu maka arahkan flow ke open gateway terdekat
        else:
          group_to_closed_gateway[i][index] = group_to_closed_gateway[i][index] -2

      else:
        temp_2_group_to_closed_gateway[i][index] = group_to_closed_gateway[i][index] -1
        group_to_closed_gateway[i][index] = group_to_closed_gateway[i][index] -1


  count_closed_gateway = 0
  count_open_gateway = 0

  changed_value = 0
  for i in range(len(detailed_array)):
    if(i == 0):
      detailed_array[i]['incoming_flow'] = 'start event'
      detailed_array[i]['outgoing_flow'] = i+1
    elif(i == len(detailed_array)-1 and detailed_array[i].get('sign') != 'end' and 'back_to' not in detailed_array[i]): # ini untuk proses terakhir namun tidak ada kata "proses berakhir")
      detailed_array[i]['incoming_flow'] = i-1
      detailed_array[i]['outgoing_flow'] = -1
    else:
      detailed_array[i]['incoming_flow'] = i-1
      detailed_array[i]['outgoing_flow'] = i+1
      if('back_to' in detailed_array[i] and 'gateway' not in detailed_array[i]):
        detailed_array[i]['outgoing_flow'] = detailed_array[i]['back_to']
      elif(detailed_array[i].get('sign') == 'end'):
        if('gateway' in detailed_array[i-1]):
          for outflow in detailed_array[i-1]['outgoing_flow']:
            if(detailed_array[outflow].get('sign') == 'end'):
              changed_index = detailed_array[i-1]['outgoing_flow'].index(outflow)
              changed_value = outflow
              detailed_array[i-1]['outgoing_flow'][changed_index] = -1
        else:
          detailed_array[i-1]['outgoing_flow'] = -1
          del detailed_array[i]['outgoing_flow']
          del detailed_array[i]['incoming_flow']
      else:
        if(detailed_array[i].get('gateway') == 'closed gateway'):
          detailed_array[i]['incoming_flow'] = group_to_closed_gateway[count_closed_gateway]
          count_closed_gateway += 1
        elif(detailed_array[i].get('gateway') == 'open gateway'):
          for ind in range(len(group_after_open_gateway[count_open_gateway])):
            if(detailed_array[group_after_open_gateway[count_open_gateway][ind]].get('titik_temu_proses')):
              group_after_open_gateway[count_open_gateway][ind] = group_after_open_gateway[count_open_gateway][ind]-1
              detailed_array[i]['outgoing_flow'] = group_after_open_gateway[count_open_gateway]
            else:
              detailed_array[i]['outgoing_flow'] = group_after_open_gateway[count_open_gateway]
          count_open_gateway += 1


  for data in group_after_open_gateway:
    first_value = data[0]
    if(first_value == -1): #jalur pertama mengarah ke end
      first_value = changed_value
    for j, nomor_index in enumerate(data):
      if(j != 0):
        if 'gateway' not in detailed_array[nomor_index]:
          detailed_array[nomor_index]['incoming_flow'] = first_value - 1 # incoming flow pada proses jika 1 dan 2

  for data in temp_2_group_to_closed_gateway:
    last_value = data[-1]
    for index in data:
        detailed_array[index]['outgoing_flow'] = last_value + 1



  for data in detailed_array:
    if(data.get('gateway') == 'open gateway'):
      if(type(data['outgoing_flow']) == int):
        data['outgoing_flow'] = [data['outgoing_flow']]
    elif(data.get('gateway') == 'closed gateway'):
      if(type(data['incoming_flow']) == int):
        data['incoming_flow'] = [data['incoming_flow']]

  print("hasil penambahan flow :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")

  index_removed = remove_unused_process(detailed_array, temp_group_to_closed_gateway, index_back_process)
  detailed_array = [element for index, element in enumerate(detailed_array) if index not in index_removed]

  print("hasil eliminasi array yang tidak digunakan :")
  for index,a in enumerate(detailed_array):
      print(str(index)+' '+str(a))
  print("")
  print("=============================")

  actors = []
  for index, data in enumerate(detailed_array):
    if(data.get('structure')):
      actors.append(data['structure']['subject'])

  actors = list(set(actors))


  actors_sequence = []

  for data in detailed_array:
    actors_sequence.append(data['structure']['subject'])

  actors = list(OrderedDict.fromkeys(actors_sequence))


  return code_to_xml(actors, detailed_array)

def change_position_y(temp_start, temp_to, tags_with_id, detailed_array):
  position_will_changed = []
  if(temp_to): # temp_to ditujukan sampai mana process kedua if bertemu gateway converging, jika nilainya ada maka ada cconverging, jika tidak berarti proses if tidak ada converging
    for start_id, to_id in zip(temp_start, temp_to):
      start_index = next(index for index, tag in enumerate(tags_with_id) if tag['id'] == start_id)
      to_index = next(index for index, tag in enumerate(tags_with_id) if tag['id'] == to_id)
      # print(start_index, to_index)

      for index in range(start_index, to_index + 1):
        position_will_changed.append(tags_with_id[index]['id'])
      # print(position_will_changed)
  else:
    for start_i in temp_start:
      start_index = next(index for index, tag in enumerate(tags_with_id) if tag['id'] == start_i)

      for index in range(start_index, len(detailed_array)):
        position_will_changed.append(tags_with_id[index]['id'])

  return position_will_changed


def code_to_xml(actors, detailed_array):
  header = '''<?xml version="1.0"?>
  <definitions xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" id="_2024041506798" targetNamespace="http://www.bizagi.com/definitions/_2024041506798" xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
  '''
  footer = '\n</definitions>'


  '========================================================================================================================'


  header_actor = '  <collaboration id="collab_actor" name="Diagram BPMN">\n'
  footer_actor = '  </collaboration>\n'
  actor = ''
  for index, act in enumerate(actors):
    actor += '    <participant id="act_' + str(index) + '" name="'+ act +'" processRef="process_act_'+ str(index) +'"></participant>\n'


  '========================================================================================================================'


  process = ''
  for index, act in enumerate(actors):
    process += '  <process id="process_act_'+ str(index) +'" name="'+ act +'">\n'
    process += '    <documentation />\n'

    for data in detailed_array:
      if(data.get('structure')):
        if(act == data['structure']['subject']):
          if(data.get('incoming_flow') or data.get('incoming_flow') == 0 and data.get('outgoing_flow')):
            if(data['incoming_flow'] == 'start event'):
              # add start event
              process += '    <startEvent id="start">\n'
              process += '      <outgoing>flow_' + str(data['index'])+ '</outgoing>\n'
              process += '    </startEvent>\n'

              # process += '    <sequenceFlow id="flow_'+ str(data['index']) +'" sourceRef="start" targetRef=""></sequenceFlow>'

              #add process
              process += '    <task id="task_'+ str(data['index']) +'" name="'+data['structure']['verb'] + ' ' + str(data['structure']['object']) +'">\n'
              process += '      <incoming>flow_'+ str(data['incoming_flow']) +'</incoming>\n'
              process += '      <outgoing>flow_'+ str(data['outgoing_flow']) +'</outgoing>\n'
              process += '    </task>\n'
            elif(data.get('gateway')):
              if(data['gateway'] == 'open gateway'):
                if(-1 in data['outgoing_flow']):
                  process += '    <endEvent id="flow_-1">\n'
                  process += '      <incoming>flow_'+ str(data['index']) +'</incoming>\n'
                  process += '    </endEvent>\n'
                process += '    <exclusiveGateway id="gateway_'+str(data['index'])+'" gatewayDirection="Diverging">\n'
                process += '      <incoming>flow_'+ str(data['incoming_flow']) +'</incoming>\n'
                for out in data['outgoing_flow']:
                  process += '      <outgoing>flow_'+ str(out) +'</outgoing>\n'
                process += '    </exclusiveGateway>\n'

              elif(data['gateway'] == 'closed gateway'):
                process += '    <exclusiveGateway id="gateway_'+str(data['index'])+'" gatewayDirection="Converging">\n'
                for incoming in data['incoming_flow']:
                  process += '      <incoming>flow_'+str(incoming)+'</incoming>\n'
                process += '      <outgoing>flow_'+str(data['outgoing_flow'])+'</outgoing>\n'
                process += '    </exclusiveGateway>\n'
            else:
              process += '    <task id="task_'+ str(data['index']) +'" name="'+data['structure']['verb'] + ' ' + str(data['structure']['object']) +'">\n'
              process += '      <incoming>flow_'+ str(data['incoming_flow']) +'</incoming>\n'
              process += '      <outgoing>flow_'+ str(data['outgoing_flow']) +'</outgoing>\n'
              process += '    </task>\n'
              if(data['outgoing_flow'] == -1):
                process += '    <endEvent id="flow_-1">\n'
                process += '      <incoming>flow_'+ str(data['index']) +'</incoming>\n'
                process += '    </endEvent>\n'
    # sequence_flow = create_sequenceFlow(process)
    # process += sequence_flow

    if(index != len(actors)-1):
      process += '  </process>\n'



  '========================================================================================================================'

  soup = BeautifulSoup(process, 'html.parser')
  process_tag = soup.find_all('process')
  sequence_flow = ''

  child_process_with_id = []
  for index, tag in enumerate(process_tag):
    child_process = tag.findChildren(recursive=False)
    for child in child_process:
      if(child.get('id')):
        child_process_with_id.append(child)


  for event in child_process_with_id:
    temp = event.find_all('outgoing')
    for outgoing in temp:
      outgoing_num = outgoing.text.split("_")[-1]
      for data in child_process_with_id:
        if('-1' in data['id']):
          if(outgoing.text == data['id']):
            sequence_flow += '    <sequenceFlow id="flow_event_'+event['id']+'_'+data['id']+'" sourceRef="'+event['id']+'" targetRef="'+data['id']+'">\n'
            sequence_flow += '    </sequenceFlow>\n'
        else:
          id_num = data['id'].split("_")[-1]
          if(outgoing_num.isdigit() and id_num.isdigit()):
            if(outgoing_num == id_num):
              if 'gatewaydirection' in event.attrs and event.attrs['gatewaydirection'].lower() == 'diverging':
                  item_detailed_aray = [item for item in detailed_array if item['index'] == int(id_num)]
                  if(item_detailed_aray):
                    if('case' in item_detailed_aray[0]):
                      item_detailed_aray[0]['case'] = item_detailed_aray[0]['case'].replace('jika','') # menggunakan [0] karena item detailed array isinya adalah list dan pasti 1
                      sequence_flow += '    <sequenceFlow id="flow_event_'+event['id']+'_'+data['id']+'" sourceRef="'+event['id']+'" targetRef="'+data['id']+'" name="'+item_detailed_aray[0]['case']+'">\n'
                    elif('referenced_case' in item_detailed_aray[0]):
                      item_detailed_aray[0]['referenced_case'] = item_detailed_aray[0]['referenced_case'].replace('jika','') # menggunakan [0] karena item detailed array isinya adalah list dan pasti 1
                      sequence_flow += '    <sequenceFlow id="flow_event_'+event['id']+'_'+data['id']+'" sourceRef="'+event['id']+'" targetRef="'+data['id']+'" name="'+item_detailed_aray[0]['referenced_case']+'">\n'
                    sequence_flow += '    </sequenceFlow>\n'
              else:
                sequence_flow += '    <sequenceFlow id="flow_event_'+event['id']+'_'+data['id']+'" sourceRef="'+event['id']+'" targetRef="'+data['id']+'">\n'
                sequence_flow += '    </sequenceFlow>\n'

  process += sequence_flow

  process += '  </process>\n'

  '========================================================================================================================'

  header_style =  ' <BPMNDiagram id="DiagramBPMN" xmlns="http://www.omg.org/spec/BPMN/20100524/DI">\n'
  header_style += '   <BPMNPlane id="DiagramElement" bpmnElement="collab_actor">\n'
  header_style += '     <extension xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'


  footer_style  = '   </BPMNPlane>\n'
  footer_style += ' </BPMNDiagram>\n'

  actor_position = ''
  height_plane = 350
  y = 0
  for index, act in enumerate(actors):
    actor_position += '     <BPMNShape id="DiagramElement_'+ act + '_' + str(index) +'" bpmnElement="act_' + str(index) + '" isHorizontal="true">\n'
    actor_position += '        <extension xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
    actor_position += '        <Bounds x="0" y="'+ str(y) +'" width="1080" height="'+ str(height_plane) +'" xmlns="http://www.omg.org/spec/DD/20100524/DC"/>\n'
    actor_position += '     </BPMNShape>\n'
    y+=350


  '========================================================================================================================'


  soup = BeautifulSoup(process, 'html.parser')
  tags_with_id = soup.find_all(attrs={"id": True})

  width_plane = 1080
  process_position = ''
  start_position = {'x': 0, 'y' :125}
  for tag in tags_with_id:
    if(tag.name != 'sequenceflow'):
      process_position +=  '     <BPMNShape id="DiagramElement_'+ tag['id'] +'" bpmnElement="'+tag['id'] + '" isHorizontal="true">\n'
      process_position += '        <extension xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
      if('start' in tag['id'] or '-1' in tag['id']):
        process_position += '        <Bounds x="'+str(start_position['x'])+'" y="'+ str(start_position['y']) +'" width="30" height="30" xmlns="http://www.omg.org/spec/DD/20100524/DC"/>\n'
      elif('process' in tag['id']): # pemisah lane actor
        if(int(tag['id'].split("_")[-1]) > 0):
          start_position['y'] += 350
          start_position['x'] = 230
          process_position += '        <Bounds x="'+str(start_position['x'])+'" y="'+ str(start_position['y']) +'" width="30" height="30" xmlns="http://www.omg.org/spec/DD/20100524/DC"/>\n'
      elif('gateway' in tag['id']):
        process_position += '        <Bounds x="'+str(start_position['x'])+'" y="'+ str(start_position['y']) +'" width="30" height="30" xmlns="http://www.omg.org/spec/DD/20100524/DC" sign="'+tag.attrs['gatewaydirection'].lower()+'"/>\n'
      else:
        process_position += '        <Bounds x="'+str(start_position['x'])+'" y="'+ str(start_position['y']) +'" width="90" height="60" xmlns="http://www.omg.org/spec/DD/20100524/DC"/>\n'
      process_position += '     </BPMNShape>\n'

      start_position['x'] += 200
      temp = start_position['x'] + 130
      if(temp >= width_plane):

        y=0
        for index, act in enumerate(actors):
          if(len(actors) == 1):
            width_plane += 150
          else:
            if index != 0:
              width_plane += 150


  if(width_plane > 1080):
    actor_position =  '   '
    for index, act in enumerate(actors):
      actor_position += '      <BPMNShape id="DiagramElement_'+ act + '_' + str(index) +'" bpmnElement="act_' + str(index) + '" isHorizontal="true">\n'
      actor_position += '        <extension xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
      actor_position += '        <Bounds x="0" y="'+ str(y) +'" width="'+str(width_plane)+'" height="'+ str(height_plane) +'" xmlns="http://www.omg.org/spec/DD/20100524/DC"/>\n'
      actor_position += '      </BPMNShape>\n'
      y += 350



  '========================================================================================================================'

  soup1 = BeautifulSoup(process_position, 'html.parser')

  tags_with_id1 = soup1.find_all(attrs={"id": True})

  temp_tasks = []
  for tag in tags_with_id1:
    if('gateway' in tag.attrs['bpmnelement']):
      bounds_tag = tag.find('bounds')
      if(bounds_tag['sign'] == 'diverging'):
        gateway_tags = soup.find_all(lambda tag: tag.has_attr('id') and 'gateway' in tag['id']) # cari tag gateway
        temp_tasks = []
        for tag in gateway_tags:
          outgoing_tags = tag.find_all('outgoing')
          for index, outgoing_tag in enumerate(outgoing_tags):
            if(index != 0):
              temp_tasks.append(outgoing_tag.string.split("_")[-1]) # ambil nomernya saja

  temp_start = []
  temp_to = []
  for tag in tags_with_id:
    if tag.name == 'exclusivegateway' and tag['gatewaydirection'] == 'Converging' :
      temp_to.append(tag['id'])


  for task in temp_tasks:
    for tag_event in tags_with_id:
      if('task_'+task == tag_event['id']):
        temp_start.append('task_'+task)
      elif('gateway_'+task == tag_event['id']):
        temp_start.append('gateway_'+task)
  change_position = change_position_y(temp_start, temp_to, tags_with_id, detailed_array)

  for tag in tags_with_id1:
    for task in change_position:
      if('task' in tag.attrs['bpmnelement']):
        if(task == tag.attrs['bpmnelement']):
          bounds_tag = tag.find('bounds')
          bounds_tag['y'] = str(int(bounds_tag['y']) + 100)


  bpmnshape_tags = soup1.find_all('bpmnshape')
  bounds_tags = soup1.find_all('bounds')

  for tag in bpmnshape_tags:
    tag.name = 'BPMNShape'
    if 'bpmnelement' in tag.attrs:
      tag.attrs['bpmnElement'] = tag.attrs.pop('bpmnelement')
      tag.attrs['isHorizontal'] = tag.attrs.pop('ishorizontal')

  for tag in bounds_tags:
    tag.name = 'Bounds'

  process_position = str(soup1.prettify())


  '========================================================================================================================'

  soup_sequence_flow = BeautifulSoup(sequence_flow, 'html.parser')
  tags_with_id_flow = soup_sequence_flow.find_all(attrs={"id": True})

  soup1 = BeautifulSoup(process_position, 'html.parser')
  flow_position = ''

  for tag in tags_with_id_flow:
    source_position = soup1.find(bpmnelement=tag['sourceref'])
    target_position = soup1.find(bpmnelement=tag['targetref'])
    source_position_y = source_position.find('bounds').get('y')
    target_position_y = target_position.find('bounds').get('y')
    source_position_x = source_position.find('bounds').get('x')
    target_position_x = target_position.find('bounds').get('x')

    flow_position += '    <BPMNEdge id="DiagramElement_'+tag['id']+'" bpmnElement="'+tag['id']+'">\n'
    flow_position += '      <extension xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'

    if(int(source_position_y) < int(target_position_y)):
      if(int(source_position_x) == int(target_position_x)):
        flow_position += '      <waypoint x="'+source_position_x+'" y="'+source_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
        flow_position += '      <waypoint x="'+target_position_x+'" y="'+target_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
      else:
        if('gateway' in tag['sourceref']):
          flow_position += '      <waypoint x="'+str(int(source_position_x)+30)+'" y="'+source_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+str(int(source_position_x)+30)+'" y="'+str(int(target_position_y)+30)+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+target_position_x+'" y="'+str(int(target_position_y)+30)+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
        else:
          flow_position += '      <waypoint x="'+source_position_x+'" y="'+source_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+source_position_x+'" y="'+target_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+target_position_x+'" y="'+target_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
    elif(int(source_position_y) > int(target_position_y)):
      if(int(source_position_x) == int(target_position_x)):
        flow_position += '      <waypoint x="'+source_position_x+'" y="'+source_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
        flow_position += '      <waypoint x="'+target_position_x+'" y="'+target_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
      else:
        if('gateway' in tag['sourceref']):
          flow_position += '      <waypoint x="'+str(int(source_position_x)-30)+'" y="'+source_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+str(int(source_position_x)-30)+'" y="'+str(int(target_position_y)-30)+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+target_position_x+'" y="'+str(int(target_position_y)-30)+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
        else:
          flow_position += '      <waypoint x="'+source_position_x+'" y="'+source_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+source_position_x+'" y="'+target_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
          flow_position += '      <waypoint x="'+target_position_x+'" y="'+target_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
    else:
      flow_position += '      <waypoint x="'+source_position_x+'" y="'+source_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
      flow_position += '      <waypoint x="'+target_position_x+'" y="'+target_position_y+'" xmlns="http://www.omg.org/spec/DD/20100524/DI" />\n'
    flow_position += '    </BPMNEdge>\n'



  '========================================================================================================================'


  soup_flow_position = BeautifulSoup(flow_position, 'html.parser')

  # Dictionary untuk melacak waypoint yang sudah ditemukan
  waypoints = set()

  bpmnedges = soup_flow_position.find_all('bpmnedge')
  coordinates = {}
  for bpmnedge in bpmnedges:
      waypoints = bpmnedge.find_all('waypoint')
      coordinates[bpmnedge['id']] = [(int(waypoint['x']), int(waypoint['y'])) for waypoint in waypoints]

  checked_pairs = set()
  for bpmnedge_id, coords in coordinates.items():
    for other_bpmnedge_id, other_coords in coordinates.items():
      if bpmnedge_id != other_bpmnedge_id and coords[0] == other_coords[0] and coords[1][0] == other_coords[1][0]:
        pair_key = tuple(sorted([bpmnedge_id, other_bpmnedge_id]))
        if pair_key not in checked_pairs:
          checked_pairs.add(pair_key)
          flow_position = flow_position.replace(f'<waypoint x="{other_coords[1][0]}" y="{other_coords[1][1]}" xmlns="http://www.omg.org/spec/DD/20100524/DI" />', f'<waypoint x="{other_coords[2][0]}" y="{other_coords[0][1]}" xmlns="http://www.omg.org/spec/DD/20100524/DI" />')




  result = header + header_actor + actor + footer_actor + process + header_style + actor_position + process_position + flow_position + footer_style + footer

  return result


def create_meeting_notes(path):
    # path_file = convert_path(path)
    # print("ini path file")
    # print(path_file)
    text = model.transcribe(path)
    # printing the transcribe
    data = text['text']
    data = data.strip().replace(',','').lower()

    wib = pytz.timezone('Asia/Jakarta')
    now_wib = datetime.now(wib)
    formatted_date_time = now_wib.strftime("%d-%m-%Y %H.%M.%S.%f")[:-3]

    meeting_notes = generate_meeting_notes(formatted_date_time+'.pdf', data)

    for i in range(len(meeting_notes)):
        if('proses dimulai dari' in meeting_notes[i]):
            temp = meeting_notes[i].find('proses dimulai dari')
            meeting_notes[i] = meeting_notes[i][temp:]
            meeting_notes[i] = meeting_notes[i].replace("proses dimulai dari", "").strip()

        elif('proses dimulai saat' in meeting_notes[i]):
            temp = meeting_notes[i].find('proses dimulai saat')
            meeting_notes[i] = meeting_notes[i][temp:]
            meeting_notes[i] = meeting_notes[i].replace("proses dimulai saat", "").strip()

    for index, story_desc in enumerate(meeting_notes):
        res = generate_xml(story_desc)
        with open('proses_'+str(index)+'.bpmn', 'w') as f:
            f.write(res)



def main():
    # Membuat jendela utama dengan tema
    root = ThemedTk(theme="arc")
    root.title("BPMN Generator")
    root.geometry("400x200")

    selected_file = [None]

    # Label dan tombol untuk mengimpor file
    label_intro = ttk.Label(root, text="Import File MP4:")
    label_intro.pack(pady=10)

    button_import = ttk.Button(root, text="Pilih File", command=lambda: import_file(label_status, selected_file))
    button_import.pack(pady=5)

    
    # Tombol untuk memulai pemrosesan
    button_start = ttk.Button(root, text="Mulai Proses", command=lambda: start_processing(label_status, selected_file, progressbar))
    button_start.pack(pady=5)

    # Label untuk status dan progress bar
    label_status = ttk.Label(root, text="Belum ada file dipilih")
    label_status.pack(pady=10)

    progressbar = ttk.Progressbar(root, mode='indeterminate')
    progressbar.pack(pady=5, fill=tk.X, padx=20)

    # Menjalankan aplikasi
    root.mainloop()

if __name__ == "__main__":
    main()