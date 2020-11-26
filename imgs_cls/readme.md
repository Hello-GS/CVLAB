一、config.py 文件中重要参数意义：
1.  val_resume:指定需要继续训练的checkpoint的路径；
2. evaluate:是否只对模型进行验证；
3. dataset:指定数据集路径
4. submit_example:指定用于存储label真值的文件
5. log_dir:指定参数记录文件夹
6. submits:指定输出结果的路径
7. bs:设定batch_size大小
8. lr:设定学习率大小
9. epochs:设定运行epoch数的大小
10  input_size:设定输入图片尺寸
11.  num_classes:分类结果数
12. dropout:指定dropout率
13. gpu_id:指定gpu_id
14. model_name:指定模型名字
15. optim: 指定优化方法
16. loss_func:指定损失函数
17. lr_scheduler: 指定学习率的scheduler

二、 重要代码文件说明：
1、main.py：进行模型训练和估计，训练保存的checkpoint在 configs.checkpoints/configs.model_name的路径下。
2、test.py：输出结果的文件，会生成训练好的模型的输出结果，输出结果保存在data文件夹下。
3、count_metric.py：计算按图片计算和按事件计算的recall、precision和f1 score。输入文件保存在data文件夹下。
4、generate_gallery.py：生成test和train数据集下的图片的特征向量，保存为query_CME.json和galllery_CME.json。
5、image_to_incident.py：输入为图片的特征向量文件，输出为事件的特征向量文件，保存为query.json和 gallery.json。
6、generate_similar_event.py：输入为事件的特征向量文件，即query.json和gallery.json，输出为test中的事件对应的五个最相似的train中的事件，保存为similar_event.json 和 similar_event.csv。
7、find_k_similar_event.py：输入为test集中的一个事件，输出为和train集中和它最相似的五个事件的图像。