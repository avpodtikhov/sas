import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import base64

class Model:
    def fit(self, trainCSV):
        df = pd.read_csv(trainCSV, sep=';')
        x_train = df.copy()
        x_train = x_train.drop("Target1", axis=1)
        x_train = x_train.drop('Target2', axis=1)
        y_phone = df['Target2']
        y_email = df['Target1']
        self.__le = LabelEncoder()
        x_train = x_train.apply(self.__le.fit_transform)
        X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(
            x_train, y_phone, test_size=0.33, random_state=42)
        X_train_em, X_test_em, y_train_em, y_test_em = train_test_split(
            x_train, y_email, test_size=0.33, random_state=42)

        clf_tree_em = DecisionTreeClassifier(max_depth=11, random_state=17)
        clf_tree_em.fit(X_train_em, y_train_em)
        clf_tree_ph = DecisionTreeClassifier(max_depth=11, random_state=17)
        clf_tree_ph.fit(X_train_ph, y_train_ph)
        self.__models = (clf_tree_em, clf_tree_ph)
        self.__save()

    def __save(self):
        with open('./model/model_email.pickle', 'wb') as f:
            pickle.dump(self.__models[0], f)

        with open('./model/model_phone.pickle', 'wb') as f:
            pickle.dump(self.__models[1], f)

        with open('./model/encoders.pickle', 'wb') as f:
            pickle.dump(self.__le, f)
    
    def load_prev(self):
        with open('./model/model_email.pickle', 'rb') as f:
            model_email = pickle.load(f)

        with open('./model/model_phone.pickle', 'rb') as f:
            model_phone = pickle.load(f)

        self.__models = (model_email, model_phone)

        with open('./model/encoders.pickle', 'rb') as f:
            self.__le = pickle.load(f)
        
    def apply(self, testCSV):
        test=pd.read_csv(testCSV, sep=';')
        id = test['ID']
        test = test.apply(self.__le.fit_transform)
        res_em = self.__models[0].predict_proba(test)
        res_ph = self.__models[1].predict_proba(test)
        d = {'ID': id, 'Target1': res_em[:,1], 'Target2': res_ph[:,1]}
        self.__test0 = pd.DataFrame(d)
        return
    
    def optimize(self, ph_cost, em_cost, sum_to_spend, withSMS, withEmail):
        self.__t1 = np.array(self.__test0.sort_values(['Target1'], ascending=0).reset_index(drop = True)['Target1'])
        self.__uid1 = np.array(self.__test0.sort_values(['Target1'], ascending=0).reset_index(drop = True)['ID'])
        self.__t2 = np.array(self.__test0.sort_values(['Target2'], ascending=0).reset_index(drop = True)['Target2'])
        self.__uid2 = np.array(self.__test0.sort_values(['Target2'], ascending=0).reset_index(drop = True)['ID'])

        # на вход дают цены за смс и email и сумма на траты. 
        # em_cost, ph_cost, sum_to_spend

        # зависимость предполагаемого отклика от потраченных денег

        #sum_to_spend = 10000
        sum_to_spend = sum_to_spend
        if sum_to_spend != 0:
            self.__sum_to_spend_min = (int)(sum_to_spend / 2)
            self.__sum_to_spend_max = (int)(sum_to_spend * 3 / 2)
            self.__step = int(sum_to_spend/100)
        else:
            self.__sum_to_spend_min = (int)(0)
            self.__sum_to_spend_max = (int)(100000)
            self.__step = int(100000/100)
        #ph_cost , em_cost = 1, 2 
        self.__expensive_cost = max(ph_cost, em_cost)
        self.__cheap_cost = min(ph_cost, em_cost)
        # Добавляем в массив resp значения матожидания для графика
        opt_list_expensive, opt_list_cheap = [], []
        resp = 0
        responses = []
        percents = []
        x = []
        emails = []
        smss = []

        if not withSMS:
            self.__cost = em_cost
            for i in range (self.__sum_to_spend_min, self.__sum_to_spend_max, self.__step):
                opt_list_expensive, opt_list_cheap = [], []
                resp = 0
                resp, opt_list, min_percent = self.__one_opt_response(i, 0)
                responses.append(resp)
                x.append(i)
                percents.append(round(min_percent, 2))
                if len(opt_list)== self.__t1.shape[0]:
                    for j in range(i, self.__sum_to_spend_max, self.__step):
                        emails.append([len(opt_list), base64.b64encode(pd.DataFrame(opt_list).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                        smss.append([0, base64.b64encode(pd.DataFrame([]).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                    break
                emails.append([len(opt_list), base64.b64encode(pd.DataFrame(opt_list).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                smss.append([0, base64.b64encode(pd.DataFrame([]).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
            return emails, smss, percents, responses, x

        if not withEmail:
            self.__cost = ph_cost
            for i in range (self.__sum_to_spend_min, self.__sum_to_spend_max, self.__step):
                opt_list_expensive, opt_list_cheap = [], []
                resp = 0
                resp, opt_list, min_percent = self.__one_opt_response(i, 1)
                responses.append(resp)
                x.append(i)
                percents.append(round(min_percent, 2))
                if len(opt_list) == self.__t1.shape[0]:
                    for j in range(i, self.__sum_to_spend_max, self.__step):
                        smss.append([len(opt_list), base64.b64encode(pd.DataFrame(opt_list).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                        emails.append([0, base64.b64encode(pd.DataFrame([]).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                    break
                smss.append([len(opt_list), base64.b64encode(pd.DataFrame(opt_list).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                emails.append([0, base64.b64encode(pd.DataFrame([]).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
            return emails, smss, percents, responses, x

        for i in range (self.__sum_to_spend_min, self.__sum_to_spend_max, self.__step):
            opt_list_expensive, opt_list_cheap = [], []
            resp = 0
            resp, opt_list_expensive, opt_list_cheap, min_percent = self.__opt_response(i)
            responses.append(resp)
            x.append(i)
            percents.append(round(min_percent, 2))
            if len(opt_list_expensive) + len(opt_list_cheap) == self.__t1.shape[0]:
                for j in range(i, self.__sum_to_spend_max, self.__step):
                    if self.__expensive_cost == em_cost:
                        emails.append([len(opt_list_expensive), base64.b64encode(pd.DataFrame(opt_list_expensive).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                        smss.append([len(opt_list_cheap), base64.b64encode(pd.DataFrame(opt_list_cheap).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                    else:
                        smss.append([len(opt_list_expensive), base64.b64encode(pd.DataFrame(opt_list_expensive).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                        emails.append([len(opt_list_cheap), base64.b64encode(pd.DataFrame(opt_list_cheap).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])   
                    break
            if self.__expensive_cost == em_cost:
                emails.append([len(opt_list_expensive), base64.b64encode(pd.DataFrame(opt_list_expensive).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                smss.append([len(opt_list_cheap), base64.b64encode(pd.DataFrame(opt_list_cheap).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
            else:
                smss.append([len(opt_list_expensive), base64.b64encode(pd.DataFrame(opt_list_expensive).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])
                emails.append([len(opt_list_cheap), base64.b64encode(pd.DataFrame(opt_list_cheap).to_csv(index=False, sep=';').encode('utf-8')).decode('utf-8')])   
        return emails, smss, percents, responses, x


    def __one_opt_response(self, sum_to_spend, sms):
        if sms:
            t = self.__t2
            uid = self.__uid2
        else:
            t = self.__t1
            uid = self.__uid1
        q = 0.95
        summ, i = 0, 0
        exp = 0
        res = []
        min_percent = 1
        while i < t.shape[0] - 1 and summ  < sum_to_spend:
            summ += self.__cost # считаем траты
            res.append(uid[i]) # сохраняем индексы
            exp += t[i] # считаем матож
            i += 1
            min_percent = t[i]
        return round(exp), res, min_percent*100

    def __opt_response(self, sum_to_spend):
    #   for cost in scale:
        min_percent = 1
        q_expensive = 0.97
        q_cheap = 0.97
        resp, summ, i, j = 0, 0, 0, 0
        exp_expensive, exp_cheap = 0, 0 # матожидание
        opt_list_expensive, opt_list_cheap = [], [] # списки айди на телефоны и эмейлы  
        sum_expensive, sum_cheap = 0, 0 # суммы цен за эсмс и за емейл для сравнения
        tcheapp_cheap_id, tcheapp_expensive_id = [], []
        while i < self.__t1.shape[0] and self.__t1[i] >= q_cheap:
            if summ + sum_cheap >= sum_to_spend:
                break;
            sum_cheap += self.__cheap_cost # считаем траты
            tcheapp_cheap_id.append(self.__uid1[i]) # сохраняем индексы
            exp_cheap += self.__t1[i] # считаем матож
            i += 1
            if (self.__t1[i] < min_percent) :
                min_percent = self.__t1[i]
        while (j < self.__t2.shape[0] - 1 and i < self.__t1.shape[0] - 1 and summ < sum_to_spend): #   пока сумма меньше ожидаемых трат
            if (exp_cheap >= exp_expensive): # смотрим какое мат_ожидание больше
                summ += sum_cheap # добавляем в траты сумму группы, которую решили взять
                opt_list_cheap.extend(tcheapp_cheap_id) # добавляем айди нужной группы
                resp += exp_cheap #
                q_cheap -= 0.03
            else:
                summ += sum_expensive # все то же самое но теперь добавляем другую группу
                opt_list_expensive.extend(tcheapp_expensive_id) #
                resp += exp_expensive      #
                q_expensive -= 0.03
            exp_expensive = 0 #
            sum_expensive = 0 #
            tcheapp_expensive_id = [] #
            while j < self.__t2.shape[0] - 1 and self.__t2[j] >= q_expensive: # 
                if summ + sum_expensive >= sum_to_spend:
                    break;
                sum_expensive += self.__expensive_cost #
                tcheapp_expensive_id.append(self.__uid2[j]) #
                exp_expensive += self.__t2[j] #
                j += 1
                if (self.__t2[i] < min_percent) :
                    min_percent = self.__t2[i]
            exp_cheap = 0
            sum_cheap = 0
            tcheapp_cheap_id = []
            while i < self.__t1.shape[0] - 1 and self.__t1[i] >= q_cheap: #
                if summ + sum_cheap >= sum_to_spend:
                    break;
                sum_cheap += self.__cheap_cost #
                tcheapp_cheap_id.append(self.__uid1[i]) #
                exp_cheap += self.__t1[i] #
                i += 1
                if (self.__t1[i] < min_percent) :
                    min_percent = self.__t1[i]
        return round(resp), opt_list_expensive, opt_list_cheap, min_percent*100