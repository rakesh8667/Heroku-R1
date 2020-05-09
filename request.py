import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Home_owner_ass_tax':2065, 'House_Rent':3300, 
                            'Property_Tax':211,'Fire_Insurance':42})

print(r.json())