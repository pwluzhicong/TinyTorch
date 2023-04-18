python main.py --input_data=./data/yacht.data --num_units=32,16 \
--loss_func=square_loss --activation_func=sigmoid --batch_size=16 \
--num_epochs=50 --learning_rate=1e-2 --momentum=0.9 \
--l2_norm=1e-3 --test_ratio=0.2

python main.py --input_data=./data/BreastCancer.data --num_units=32,16,8 \
--loss_func=log_binary_loss --activation_func=sigmoid --batch_size=16 \
--num_epochs=50 --learning_rate=1e-2 --momentum=0.9 \
--l2_norm=1e-3 --test_ratio=0.2
