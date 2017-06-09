export USERNAME="jrafati"
export PASSWORD="Nima8king$"
export COMPETITION="planet-understanding-the-amazon-from-space"
kg download -u $USERNAME -p $PASSWORD -c $COMPETITION -f "test-jpg-additional.tar.7z"
kg download -u $USERNAME -p $PASSWORD -c $COMPETITION -f "test-jpg.tar.7z"
kg download -u $USERNAME -p $PASSWORD -c $COMPETITION -f "train-jpg.tar.7z"
kg download -u $USERNAME -p $PASSWORD -c $COMPETITION -f "train_v2.csv.zip"

7za x "test-jpg-additional.tar.7z"
tar xf "test-jpg-additional.tar"

7za x "test-jpg.tar.7z"
tar xf "test-jpg.tar.7z"

7za x "train_v2.csv.zip"
tar xf "train-jpg.tar"

7za x "train-jpg.tar.7z"
tar "train-jpg.tar"