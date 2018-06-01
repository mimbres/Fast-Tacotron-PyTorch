python train_text2mel -exp t2m_16BN -e 1000 -btr 16 -bn True -ss False -gen 10

python train_text2mel -exp t2m_16ss -e 1000 -btr 16 -bn False -ss True -gen 10

python train_text2mel -exp t2m_16BNss -e 1000 -btr 16 -bn True -ss True -gen 10

python train_text2mel -exp t2m_64 -e 1000 -btr 64 -bn False -ss False -gen 10