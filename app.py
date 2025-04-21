from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load EigenPlaces model
def load_eigenplaces_model():
    model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet18", fc_output_dim=512)
    num_classes = 251  # Adjust as per fine-tuned model
    model.aggregation[3] = nn.Linear(in_features=2048, out_features=num_classes)
    state_dict = torch.load("best_model95.59%.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_eigenplaces_model().to(device)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = [' Statue of Amun', '6_October_Bridge', 'Abdeen_Palace', 'Abu_Ghurab', 'Abu_Haggag_Mosque', 'Abu_Qir_Bay', 
               'Abu_el-Abbas_el-Mursi_Mosque', 'Agiba_beach', 'Aguilkia_Island', 'Ahmed_Shawki_Museum', 'Akhenaten', 'Al-Aqmar_Mosque', 'Al-Ashraf_Mosque', 'Al-Azhar_Park_(Cairo)', 'Al-Fath_Mosque', 'Al-Ghuri_Complex', 'Al-Jawhara_Palace_museum', 'Al-Manyal_Palace_Museum', 'Al-Nur_Mosque', 'Al-Qurn', "Al-Rifa'i_Mosque", "Al-Salih_Tala'i_Mosque", 'Al-Sayeda_Nafeesah_Mosque', 'Al-Sayeda_Zainab_Mosque', "Al-Shate'e_Mosque", 'Al_Fattah_Al_Alim_Mosque_(Cairo)', 'Alexandria_National_Museum', 'Alexandria_Opera_House', 'Alexandria_Port', 'Alexandria_Stadium', 'Alexandria_Zoo', 'Amada', 'AmenhotepIII', 'Amir_Taz_Palace', 'Amr_Ibn_al-Aas_Mosque_(Damietta)', 'Amun Relief', 'Another Statue of Amenhotep iii', 'Antoniadis_Palace', 'Aqsunqur_Mosque', 'Aquarium_Grotto_Garden', 'Aswan_Botanical_Garden', 'Aswan_High_Dam', 'Aswan_Low_Dam', 'Aswan_Museum', 'Attarine_Mosque', 'Bab_Zuwayla', 'Bab_al-Futuh', 'Bab_al-Nasr_(Cairo)', 'Bagawat', 'Bahr_Yousef_canal', 'Baron_Empain_Palace', 'Bayt_Al-Suhaymi', 'Ben_Ezra_Synagogue', 'Beni_Hassan', 'Bent_Pyramid', 'Bibliotheca_Alexandrina', 'Bibliotheca_Alexandrina_planetarium', 'Black_Pyramid_of_Amenemhat_III', 'Blue_Hole_(Red_Sea)', 'Borg_El_Arab_Stadium', 'Cairo_Citadel', 'Cairo_International_Stadium', 'Cairo_Opera_House', 'Cathedral_of_St._Mark,_Alexandria', "Children's_Civilisation_and_Creativity_Centre", 'Citadel_of_Qaitbay', 'City_of_the_dead_(Cairo)', 'Cleopatra', 'Cleopatra Relief', 'Collections_of_the_Imhotep_Museum_in_Saqqara', 'Colossi_of_Memnon', 'Coloured_Canyon', 'Coptic_Museum_in_Cairo', 'Corniche_(Alexandria)', 'Crystal_Mountain,_White_Desert', 'Deir_el-Bahari', 'Deir_el-Medina', 'Deir_el-Muharraq', 'Deir_el-Qadisa_Damyana', 'Dendera_Temple_complex', 'Desouk_Stadium', 'Dra_Abu_el-Naga', 'Dream_Park', 'EL-Safa_Park', 'Edfu_Temple', 'Egyptian_Geological_Museum', 'Egyptian_Museum_(Cairo)', 'Egyptian_National_Library', 'Egyptian_National_Military_Museum', 'El-Bahr_mosque,_Dumyat', "El-Ma'eini_mosque,_Dumyat", 'El_Alamein_Military_Museum', 'El_Ferdan_Railway_Bridge', 'Elephantine', 'Eliyahu_Hanavi_Synagogue_(Alexandria)', 'Emir_Qurqumas_complex', 'Esna_Temple', 'Faiyum_Zoo', 'Famine_stele', 'Fatima_Khatun_Mosque', 'Fatimid_Cemetery_in_Aswan', 'Garden_City,_Cairo', 'Gayer-Anderson_Museum', 'Gebel_el-Silsila', 'Gebel_el-Teir,_el-Kharga', 'Gezira', 'Gezira_Center_for_Modern_Art', 'Gidi_Pass', 'Gilf_Kebir_Plateau', 'Giyushi_Mosque,_Cairo', 'Giza_Plateau', 'Giza_Zoo', 'Giza_pyramid_complex', 'Goddess Isis with her child', 'Grand_Egyptian_Museum', 'Great_Hypostyle_Hall_of_Karnak', 'Great_Pyramid_of_Giza', 'Great_Sphinx_of_Giza', 'Great_Temple_of_the_Aten', 'Greco-Roman_Museum,_Alexandria', 'Greek_Orthodox_Cathedral_of_Evangelismos,_Alexandria', 'Green_Island_(Egypt)', 'Hanging_Church_(Cairo)', 'Hatem_Mosque', 'Hatshepsut', 'Hatshepsut face', 'Heliopolis_Palace', 'Hurghada_Grand_Aquarium', 'Islamic_Cairo', 'Island_of_Bigeh', 'KV17', 'KV62', 'Karnak_precinct_of_Amun-Ra', 'Khan_el-Khalili', 'Khayrbak_Mosque', 'King Thutmose III', 'Kiosk_of_Qertassi', 'Kiosk_of_Trajan_in_Philae', "Kitchener's_Island", 'Kom_el-Shoqafa', 'Koubbeh_Palace', 'Lake_Timsah', 'Layer_Pyramid', 'Luxor_Museum', 'Luxor_Temple', 'Madrasah_of_Al-Nasir_Muhammad', 'Madrasah_of_Sarghatmish', 'Mahmoud_Khalil_Museum', 'Mallawi_Museum', 'Mask of Tutankhamun', "Mastabat_al-Fir'aun", 'Mausoleum_of_Aga_Khan', 'Medinet_Madi', 'Meidum', 'Mohandessin', 'Mokattam', 'Monastery_of_Saint_Anthony', 'Monastery_of_Saint_Bishoy', 'Monastery_of_Saint_Macarius_the_Great', 'Monastery_of_Saint_Samuel_the_Confessor', 'Monastery_of_Saint_Simeon_in_Aswan', 'Montaza_Palace', 'Mortuary_Temple_of_Hatshepsut', 'Mortuary_Temple_of_Seti_I_in_Qurna', 'Mortuary_Temple_of_Thutmosis_III', 'Mosque-Madrassa_of_Sultan_Hassan', 'Mosque_of_Ibn_Tulun', 'Mosque_of_Qajmas_al-Ishaqi', 'Mosque_of_Qanibay_al-Muhammadi', 'Mosque_of_Qanibay_al-Rammah', 'Mosque_of_Saint_Ibrahim_El-Desouky', 'Mosque_of_Sultan_Abu_al-Ila', 'Mosque_of_Sultan_al-Muayyad', 'Mosque_of_Sultan_al-Zahir_Baybars', 'Mosque_of_al-Mahmudiya', 'Mosque_of_al-Maridani', 'Muhammad_Ali_Mosque', 'Muizz_Street', 'Museum_of_Constantine_P._Cavafy', 'Museum_of_Islamic_Art,_Cairo', "Na'ama_Bay", 'Nabq_Protected_Area', 'Nefertiti', 'New_Kalabsha', 'Nilometer_in_Rhoda_Island', 'Nubia_Museum,_Aswan', 'Nyuserre_sun_temple', 'Orman_Garden', 'Osireion', 'Paromeos_Monastery', 'Petrified_Forest_near_Maadi', 'Pharaon_Island', "Pompey's_Pillar,_Alexandria", 'Port_Said_Lighthouse', 'Port_Tewfik_Memorial', 'Ptolemaic_Temple_of_Hathor_in_Deir_el-Medina', 'Pyramid_of_Ahmose', 'Pyramid_of_Amenemhat_I', 'Pyramid_of_Amenemhat_III_in_Hawara', 'Pyramid_of_Ameni_Qemau', 'Pyramid_of_Baka', 'Pyramid_of_Djedefra', 'Pyramid_of_Djedkare_Isesi', 'Pyramid_of_Djoser', 'Pyramid_of_Khafra', 'Pyramid_of_Khendjer', 'Pyramid_of_Khentkaus_II', 'Pyramid_of_Menkaure', 'Pyramid_of_Merenre_Nemtyemsaf_I', 'Pyramid_of_Neferefre', 'Pyramid_of_Neferirkare', 'Pyramid_of_Nyuserre_Ini', 'Pyramid_of_Pepi_I', 'Pyramid_of_Pepi_II', 'Pyramid_of_Qakare_Ibi', 'Pyramid_of_Sahure', 'Pyramid_of_Seila', 'Pyramid_of_Senusret_I', 'Pyramid_of_Senusret_II', 'Pyramid_of_Teti', 'Pyramid_of_Unas', 'Pyramid_of_Userkaf', 'Qaed_Ibrahim_Mosque', 'Qalawun_complex', 'Qasr_Qarun', 'Qasr_al-Nil_Bridge', 'Qubbet_el-Hawa', 'Qurnet_Murai', 'Rahmanyia_island', 'Ramesseum', 'Ramses II Red Granite Statue', 'Ras_Muhammad', 'Ras_el-Tin_Palace', 'Red_Monastery', 'Red_Pyramid', 'Relations_of_Egypt_and_Italy', 'Rhoda_Island', 'Sabil_of_Abd_al-Rahman_Katkhuda', 'Sadat_museum', 'Saint_Barbara_Church_in_Coptic_Cairo', "Saint_Catherine's_Monastery,_Mount_Sinai", 'Saint_George_Church_in_Coptic_Cairo', "Saint_Mark's_Coptic_Orthodox_Cathedral,_Cairo", 'Saint_Mark_Coptic_Orthodox_Church_(Heliopolis)', 'Saints_Sergius_and_Bacchus_Church,_Cairo', 'Sakakini_Palace', 'San_Stefano_Grand_Plaza', 'Saqqara', 'Sayeda_Aisha_Mosque', 'Sehel_Island', "Sha'ar_Hashamayim_Synagogue_(Cairo)", "Sha'b_Abu_el-Nuhas", 'Shalalat_Garden,_Alexandria', "Shepheard's_Hotel", 'Siwa', 'Sohag_Stadium', 'Soma_Bay', 'Speos_Artemidos', 'Sphinx_of_Memphis', "St._Catherine's_Cathedral,_Alexandria", 'Statue of King Zoser', 'Statue of Princess Meketaton', 'Statue of Tutankhamun with Ankhesenamun', 'Suez_Canal_Bridge', 'Sulayman_Agha_al-Silahdar_Mosque', 'Sulayman_Pasha_Mosque', 'Sultan_Qaytbay_Complex', 'Synagogue_of_Moses_Maimonides', 'Syrian_Monastery', 'Temple_of_Derr', 'Temple_of_Hibis', 'Temple_of_Isis_in_Deir_el-Shelwit', 'Temple_of_Isis_in_Philae', 'Temple_of_Khonsu_in_Karnak', 'Temple_of_Kom_Ombo', 'Temple_of_Seti_I_in_Abydos', 'The Great Temple of Ramesses II', 'Theban_Necropolis', 'Tirbana_mosque,_Alexandria', 'Tomb_of_Hetepheres', 'Tomb_of_Kheruef', 'Tomb_of_Nakht_TT52', 'Tomb_of_Nefertari', 'Tomb_of_Unknown_Soldier_in_Cairo', 'Umm_Kulthum_Museum', 'Umm_el-Qaab', 'Unfinished_obelisk_in_Aswan', 'Userkaf_sun_temple', 'Valley_of_the_Golden_Mummies', 'Valley_of_the_Queens', 'WV22', 'Wadi_Degla', 'Wadi_Hammamat', 'Wadi_el-Raiyan', 'Wadi_el_Gemal_National_Park', 'White_Monastery', 'White_Pyramid_of_Amenemhat_II', 'bent pyramid of King Sneferu', 'head Statue of Amenhotep iii', 'red statue of Thutmose iii', 'standing statue of tutankhamun']


   

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({'message': 'Send a POST request with an image to get predictions'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
        
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(save_path)


        WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        wiki_response = ""
        response = requests.get(WIKIPEDIA_API_URL + predicted_class.replace(" ", "_"))
        if response.status_code == 200:
            wiki_response = response.json().get("extract")
        else:
            wiki_response = "Failed to fetch data from Wikipedia."
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence*100:.2f}%",
            'image_url': f'/Uploads/{filename}',
            'Wiki_response': wiki_response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

