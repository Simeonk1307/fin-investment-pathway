for g in $(rpk group list); do
    rpk group delete "$g"
done
