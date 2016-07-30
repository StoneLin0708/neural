#include "core/include/Layer.hpp"
#include "method/include/Method.hpp"

namespace nn {
namespace feedforward {

    class FeedForwardCalLayer : public CalLayer{
    public:
        FeedForwardCalLayer(int Layer, int Nodes, int Input, double LearningRate,
                         fun::fact_t act, fun::fact_t dact);
        virtual ~FeedForwardCalLayer(){}
        //forward
        mat weight;
        rowvec sum;

        //trainig
        double LearningRate;
        rowvec delta;
        mat wupdate;
        mat wupdates;

        void fp(rowvec *in);

    protected:

        fun::fact_t fact;
        fun::fact_t fdact;

    };

    class InputLayer : public BaseLayer{
    public:
        InputLayer(int Nodes);

    };

    class HiddenLayer : public FeedForwardCalLayer{
    public:
        HiddenLayer(int Layer, int Nodes, int Input, double LearningRate,
                 fun::fact_t act, fun::fact_t dact);
        //void operator=(const HiddenLayer&);

        void clear();
        void bp(BaseLayer *LowLayer);
        void update();

    };

    class OutputLayer : public FeedForwardCalLayer{
    public:
        OutputLayer(int Layer, int Nodes, int Input, double LearningRate,
                 fun::fact_t act, fun::fact_t dact,
                 fun::fcost_t cost, fun::fcost_t dcost
                    );
        //void operator=(const OutputLayer&);

        void clear();
        void bp(BaseLayer *LowLayer);
        void update();

        void CalCost();

        rowvec desire;

        fun::fcost_t fcost;
        fun::fcost_t fdcost;

        rowvec cost;
        rowvec costs;


    };

}
}
