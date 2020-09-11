
model.detect_diff_label_batch(diff_type='NOR', early_mean_loss=early_loss, batch_size=1024) #1.11 : pre-training 


def detect_diff_label_batch(self, diff_type='NOR', early_mean_loss=1.11, batch_size=1024): # early_mean_loss -> pre-training validation loss!
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.certain_noise_data = []
        self.uncertain_noise_data = []
        self.certain_corr_data = []
        self.uncertain_corr_data = []
        start = time.time() # 시작시간        
        print('[detect_diff_label_batch] len:', len(gd_data.inputs_tr), ' ,shape:', gd_data.inputs_tr.shape)

        start_idx = 0
        next_idx = batch_size
        while True:
            text_seq_arr = gd_data.inputs_tr[start_idx:next_idx, :]
            #print('start_idx:', start_idx, ' ,next_idx:', next_idx, ' ,text_seq_arr:', text_seq_arr.shape, ' ,len:', len(text_seq_arr))  
                      
            predicted_early = self.early_model.predict_on_batch(text_seq_arr) 
            predicted_early_max_idx_list = []
            for sub_list in predicted_early: predicted_early_max_idx_list.append(np.argmax(sub_list, axis=-1))
            
            if diff_type == 'NOR': predicted_overfit = self.overfit_model.predict_on_batch(text_seq_arr)
            else: predicted_overfit = self.overfit_model.predict_on_batch([np.array([gd_data.targets_tr_2digit[idx]]), np.array([text_seq])]) 
            predicted_overfit_max_idx_list = []
            for sub_list in predicted_overfit: predicted_overfit_max_idx_list.append(np.argmax(sub_list, axis=-1))

            for batch_idx, (predicted_early_max_idx, predicted_overfit_max_idx) in enumerate(zip(predicted_early_max_idx_list, predicted_overfit_max_idx_list)):
                #print('start_idx:', start_idx, ' ,batch_idx:', batch_idx, ' ,predicted_early_max_idx:', predicted_early_max_idx, ' ,predicted_overfit_max_idx:', predicted_overfit_max_idx)
                decoded_ealry = output_vocab_2digit.reverse_word_index[predicted_early_max_idx]
                decoded_overfit = output_vocab_2digit.reverse_word_index[predicted_overfit_max_idx]

                #predicted_overfit_max_per = round(np.max(predicted_overfit[0]), 2)
                idx  = start_idx + batch_idx
                hs = gd_data.y_train_text_2digit[idx]
                gd = gd_data.x_train_text[idx]   

                loss_value = loss(gd_data.targets_tr_2digit[idx], predicted_early[batch_idx])
                loss_value = loss_value.numpy()
                loss_value = round(loss_value, 3)

                # noise-data or 오류
                if decoded_ealry != decoded_overfit : 
                    if loss_value > early_mean_loss: self.certain_noise_data.append([hs, gd, decoded_ealry, decoded_overfit, loss_value])
                    else: self.uncertain_noise_data.append([hs, gd, decoded_ealry, decoded_overfit, loss_value])            
                else: 
                  if loss_value <= early_mean_loss: self.certain_corr_data.append([hs, gd, decoded_ealry, decoded_overfit, loss_value])
                  else: self.uncertain_corr_data.append([hs, gd, decoded_ealry, decoded_overfit, loss_value])

                if idx % 10000 == 0:
                    print('idx:', idx, ' decoded_ealry:,', decoded_ealry, ' ,decoded_overfit:', decoded_overfit, ' ,text:', gd_data.x_train_text[idx], ' loss_value:', loss_value)
                    print('certain_corr_data:', len(self.certain_corr_data), ' ,uncertain_corr_data:', len(self.uncertain_corr_data), ' ,cetain_noise_data:', len(self.certain_noise_data), ' ,uncetain_noise_data:', len(self.uncertain_noise_data))
                    print("Time taken for model fit: ", round(((time.time() - start)/3600),2), "hours.")

            start_idx = next_idx
            if start_idx >= len(gd_data.inputs_tr): break
            if (next_idx + batch_size) >= len(gd_data.inputs_tr): next_idx = len(gd_data.inputs_tr)
            else: next_idx += batch_size

        print('--detect_diff_label end--')
        print('len(self.noise_data):', len(self.noise_data), ' ,len(self.corr_data):', len(self.corr_data))
        print("Time taken for model fit: ", round(((time.time() - start)/3600),2), "hours.")

