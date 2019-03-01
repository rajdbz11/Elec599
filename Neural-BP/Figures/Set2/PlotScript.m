figure; subplot(2,1,1)
bar(1:20, r1, 0.75,'b')
subplot(2,1,2)
area(s1Vec, ph1Vec/sum(ph1Vec),'FaceColor','b');
axis([1 20 0 2])
xlabel('Neuron index')
ylabel('Activity')
title('Neural activity and posterior distribution for \phi(s_3)')
title('Neural activity and posterior distribution for \phi(s_1)')
xlabel('s_1')
ylabel('Posterior')

figure; subplot(2,1,1)
bar(1:20, r2, 0.75,'r')
subplot(2,1,2)
area(s1Vec, ph2Vec/sum(ph2Vec),'FaceColor','r');
axis([1 20 0 2])
xlabel('Neuron index')
ylabel('Activity')
title('Neural activity and posterior distribution for \phi(s_2)')
xlabel('s_2')
ylabel('Posterior')

figure; subplot(2,1,1)
bar(1:20, r3, 0.75,'g')
subplot(2,1,2)
area(s1Vec, ph3Vec/sum(ph3Vec),'FaceColor','g');
axis([1 20 0 2])
xlabel('Neuron index')
ylabel('Activity')
xlabel('s_3')
ylabel('Posterior')
title('Neural activity and posterior distribution for \phi(s_3)')
axis([-5 5 0 8e-3])
title('Neural activity and posterior distribution for \phi_1(s_1)')
title('Neural activity and posterior distribution for \phi_1(s_1)')
axis([-5 5 0 8e-3])
title('Neural activity and posterior distribution for \phi_2(s_2)')
title('Neural activity and posterior distribution for \phi_3(s_3)')


 figure; bar(1:20, r3 + M13_P + M23_P, 0.75, 'g')
hold on
bar(1:20, M13_P + M23_P, 0.75, 'r')
bar(1:20, M13_P, 0.75, 'b')

