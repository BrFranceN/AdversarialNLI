#old but gold

def opposite_isnot(filtered_dataset):

    new_samples = []


    for elem in filtered_dataset:

        new_sample = {}
        text_prem = elem['premise']
        text_hp = elem['hypothesis']
        label = elem['label']


        if label == 'CONTRADICTION':
            new_sample['premise'] = text_prem
            new_sample['hypothesis'] = text_hp.replace(" is not ", " is ")
            new_sample['label'] = 'ENTAILMENT'
            new_samples.append(new_samples)
            

        srl_pre = elem['srl']['premise'] 
        srl_hp = elem['srl']['hypothesis']

        tokens_hp = srl_hp['tokens']
        annotations_pre = srl_pre['annotations']   
        annotations_hp = srl_hp['annotations']


        

        for annotation in annotations_hp:
            print("annotation:: ",annotation)
            token_index = annotation['tokenIndex']
            verb = tokens_hp[token_index]['rawText']
            print("verb:: ",verb )


            print("PropBank:")
            propbank = annotation['englishPropbank']
            frame_name = propbank['frameName']
            print(f" Frame: {frame_name}")
            for role_info in propbank['roles']:
                role = role_info['role']
                    
                print("role:: ",role)
                span = role_info['span']
                text = get_text_from_span(tokens_hp, span)
                if role == "ARG0": 
                    print("AGENT:: ", text )
                    if (text in text_prem ):
                        print('agent found in prem!')
                elif role == "ARG1":
                    print("PATIENT::", text)
                    if(text in text_prem):
                        print('patient found in prem!')
                elif role == "ARG2":
                    print("ATTRIBUTE::", text)
                    if(text in text_prem):
                        print('attribute found found in prem!')
            print()
            

        
        
        label = elem['label']
        hp = elem['hypothesis']
        
        print('hypothesis:: ',hp)
        print('label:: ',elem['label'])

        break