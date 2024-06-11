import sqlite3


conn = sqlite3.connect('breeds.db')
cursor = conn.cursor()


cursor.execute('''
CREATE TABLE breeds(
    breed_index INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT)
''')


breeds_data = [
    (0, 'Chihuahua', 'Данная порода собак склонна к заболеваниям сердечно-сосудистой системы, эпилепсии, ожирению, аллергиям на рзные компоненты.Следует регулярно делать УЗИ сердца, особенном в пожилом возрасте. Нужно следить за питанием собаки, для сбланасированного питания и исключения аллергический реакций лучше использовать промышленные корма для мелких пород собак. Так же у породы встречаются различные проблеммы с зубами (кариес, пульпит и др.), для поддержания сздоровья зубов нужно чистить их специальной зеткой и пастой для собак. В зимний период часто встречаются случаи ОРВИ так как это порода собак не имеет подшорстка ее нужно одевать в одежду для собак.'),
    (1, 'Japanese_spaniel', 'У данной породы собак часто встречаются проблемы со стороны дыхания, это связанно с особенностью строения черепа. При  высокой температуре окружающей среды, собакам желательно предоставить прохладное место так как может произойти тепловой удар. Так же если вы заметили частое слезотечение у вашей собаки (коричневые дородки у глаза) следует обратьтиться к ветеринарному врачу потому что у данной породы собак встречается дистихиаз, проявляющийся в образовании дополнительного ряда ресниц, что может привезти к язвам, катаракте и прочему.'),
    (2, 'Maltese_dog', 'Так как у данной породы опущены уши им нужно периодически поднимать ушки для "проветривани". Это можно делать несколько раз в день буквально на пару секунд. Эта порода предрасположенно к заворту кишечника по этому нужно строго соблюдать режим кормления и выгула питомца. Часто встречается дермойдный синус - это одно или множество маленьких отверстий, которые могут иметь пучки волос, выступающих из них. Чаще встречается в районе шеи.'),
    (3, 'Pekinese', 'Описание породы Pekinese'),
    (4, 'Shih-Tzu', 'Описание породы Shih-Tzu'),
    (5, 'Blenheim_spaniel', 'Описание породы Blenheim_spaniel'),
    (6, 'papillon', 'Описание породы papillon'),
    (7, 'toy_terrier', 'Описание породы toy_terrier'),
    (8, 'Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (9, 'Afghan_hound', 'Описание породы Afghan_hound'),
    (10, 'basset', 'Описание породы basset'),
    (11, 'beagle', 'Описание породы beagle'),
    (12, 'bloodhound', 'Описание породы bloodhound'),
    (13, 'bluetick', 'Описание породы bluetick'),
    (14, 'black-and-tan_coonhound', 'Описание породы black-and-tan_coonhound'),
    (15, 'Walker_hound', 'Описание породы Walker_hound'),
    (16, 'English_foxhound', 'Описание породы English_foxhound'),
    (17, 'redbone', 'Описание породы redbone'),
    (18, 'borzoi', 'Описание породы borzoi'),
    (19, 'Irish_wolfhound', 'Описание породы Irish_wolfhound'),
    (20, 'Italian_greyhound', 'Описание породы Italian_greyhound'),
    (21, 'whippet', 'Описание породы whippet'),
    (22, 'Ibizan_hound', 'Описание породы Ibizan_hound'),
    (23, 'Norwegian_elkhound', 'Описание породы Norwegian_elkhound'),
    (24, 'otterhound', 'Описание породы otterhound'),
    (25, 'Saluki', 'Описание породы Saluki'),
    (26, 'Scottish_deerhound', 'Описание породы Scottish_deerhound'),
    (27, 'Weimaraner', 'Описание породы Weimaraner'),
    (28, 'Staffordshire_bullterrier', 'Описание породы Staffordshire_bullterrier'),
    (29, 'American_Staffordshire_terrier', 'Описание породы American_Staffordshire_terrier'),
    (30, 'Bedlington_terrier', 'Описание породы Bedlington_terrier'),
    (31, 'Border_terrier', 'Описание породы Border_terrier'),
    (32, 'Kerry_blue_terrier', 'Описание породы Kerry_blue_terrier'),
    (33, 'Irish_terrier', 'Описание породы Irish_terrier'),
    (34, 'Norfolk_terrier', 'Описание породы Norfolk_terrier'),
    (35, 'Norwich_terrier', 'Описание породы Norwich_terrier'),
    (36, 'Yorkshire_terrier', 'Описание породы Yorkshire_terrier'),
    (37, 'wire-haired_fox_terrier', 'Описание породы haired_fox_terrier'),
    (38, 'Lakeland_terrier', 'Описание породы Lakeland_terrier'),
    (39, 'Sealyham_terrier', 'Описание породы Sealyham_terrier'),
    (40, 'Airedale', 'Описание породы Airedale'),
    (41, 'cairn', 'Описание породы cairn'),
    (42, 'Australian_terrier', 'Описание породы Australian_terrier'),
    (44, 'Dandie_Dinmont', 'Описание породы Dandie_Dinmont'),
    (45, 'Boston_bull', 'Описание породы Boston_bull'),
    (46, 'miniature_schnauzer', 'Описание породы miniature_schnauzer'),
    (47, 'giant_schnauzer', 'Описание породы giant_schnauzer'),
    (48, 'standard_schnauzer', 'Описание породы standard_schnauzer'),
    (49, 'Scotch_terrier', 'Описание породы Scotch_terrier'),
    (50, 'Tibetan_terrier', 'Описание породы Tibetan_terrier'),
    (51, 'silky_terrier', 'Описание породы silky_terrier'),
    (52, 'coated_wheaten_terrier', 'Описание породы coated_wheaten_terrier'),
    (53, 'West_Highland_white_terrier', 'Описание породы West_Highland_white_terrier'),
    (54, 'Lhasa', 'Описание породы Lhasa'),
    (55, 'coated_retriever', 'Описание породы coated_retriever'),
    (56, 'curly_coated_retriever', 'Описание породы curly_coated_retriever'),
    (57, 'golden_retriever', 'Описание породы golden_retriever'),
    (58, 'Labrador_retriever', 'Описание породы Labrador_retriever'),
    (59, 'Chesapeake_Bay_retriever', 'Описание породы Chesapeake_Bay_retriever'),
    (60, 'German_short_haired', 'Описание породы German_short_haired'),
    (61, 'vizsla', 'Описание породы vizsla'),
    (62, 'English_setter', 'Описание породы English_setter'),
    (63, 'Irish_setter', 'Описание породы Irish_setter'),
    (64, 'Gordon_setter', 'Описание породы Gordon_setter'),
    (65, 'Brittany_spaniel', 'Описание породы Brittany_spaniel'),
    (66, 'clumber', 'Описание породы clumber'),
    (67, 'English_springer', 'Описание породы English_springer'),
    (68, 'Welsh_springer_spaniel', 'Описание породы Welsh_springer_spaniel'),
    (69, 'cocker_spaniel', 'Описание породы cocker_spaniel'),
    (70, 'Sussex_spaniel', 'Описание породы Sussex_spaniel'),
    (71, 'Irish_water_spaniel', 'Описание породы Irish_water_spaniel'),
    (72, 'kuvasz', 'Описание породы kuvasz'),
    (73, 'schipperke', 'Описание породы schipperke'),
    (74, 'groenendael', 'Описание породы groenendael'),
    (75, 'malinois', 'Описание породы malinois'),
    (76, 'briard', 'Описание породы briard'),
    (77, 'kelpie', 'Описание породы kelpie'),
    (78, 'komondor', 'Описание породы komondor'),
    (79, 'Old_English_sheepdog', 'Описание породы Old_English_sheepdog'),
    (80, 'Shetland_sheepdog', 'Описание породы Shetland_sheepdog'),
    (81, 'collie', 'Описание породы collie'),
    (82, 'Border_collie', 'Описание породы Border_collie'),
    (83, 'Bouvier_des_Flandres', 'Описание породы Bouvier_des_Flandres'),
    (84, 'Rottweiler', 'Описание породы Rottweiler'),
    (85, 'Doberman', 'Описание породы Doberman'),
    (86, 'German_shepherd', 'Описание породы German_shepherd'),
    (87, 'miniature_pinscher', 'Описание породы miniature_pinscher'),
    (88, 'Greater_Swiss_Mountain_dog', 'Описание породы Greater_Swiss_Mountain_dog'),
    (89, 'Bernese_mountain_dog', 'Описание породы Bernese_mountain_dog'),
    (90, 'Appenzeller', 'Описание породы Appenzeller'),
    (91, 'EntleBucher', 'Описание породы EntleBucher'),
    (92, 'boxer', 'Описание породы boxer'),
    (93, 'bull_mastiff', 'Описание породы bull_mastiff'),
    (94, 'Tibetan_mastiff', 'Описание породы Tibetan_mastiff'),
    (95, 'French_bulldog', 'Описание породы French_bulldog'),
    (96, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (97, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (98, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (99, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (100, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (101, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (102, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (103, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (104, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (105, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (106, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (107, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (108, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (109, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (110, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (111, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (112, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (113, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (114, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (115, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (116, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (117, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (118, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (119, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    (120, 'n02087394-Rhodesian_ridgeback', 'Описание породы Rhodesian_ridgeback'),
    
]

cursor.executemany('''
INSERT INTO breeds VALUES (?, ?, ?)
''', breeds_data)



conn.commit()

conn.close()
